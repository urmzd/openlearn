# Encoder-decoder Transformer: post-LN and pre-LN, Noam, label smoothing

## Overview

| | |
|---|---|
| **Module** | `L5.5` · build · Python · Pass 5 · 2 h reading |
| **Solution** | [`python/tinyllm/xfmr/transformer.py`](../../../../course/solutions/python/tinyllm/xfmr/transformer.py): `TransformerConfig`, `EncoderLayer`, `DecoderLayer`, `Transformer` (`encode`, `decode`, `decode_step`), `label_smoothed_loss`, `noam_rate`, `make_optimizer`, `fit`, `translate`, `save_transformer`, `load_transformer`; model oracle tests in [`learner-tests/L5.5/.../test_transformer.py`](../../../../course/solutions/learner-tests/L5.5/python/tests/l5-5-transformer/test_transformer.py) |
| **Contract** | [`course/contracts/py/tinyllm/xfmr/transformer.pyi`](../../../../course/contracts/py/tinyllm/xfmr/transformer.pyi) |
| **Needs** | `L5.2` masks · `L5.3` attention · `L5.4` `SinusoidalPE` · `L0.1` · `L0.2` · `L0.3` `cross_entropy` · `L0.4` layers · `L0.5` `DataLoader`, `train_step` · `L0.6` safetensors · `M10.3` `Adam` · `M10.4` `noam` · `M07.3` `normal_init` · `M06.3` `PCG32` · `L4.4` `beam_search` |
| **Used by** | `L6.7` the zoo's `transformer` rows on the addition task. This is where Part 5 comes together: exact match 0.98 on unseen sums, and a demo where post-LN fails without warmup while pre-LN trains |
| **Optional depth** | Vaswani et al. (2017), sections 3 and 5; Xiong et al., "On Layer Normalization in the Transformer Architecture" (ICML 2020); Szegedy et al. (2016), section 7 (label smoothing); Rush, *The Annotated Transformer* |

## Key Takeaways

- The encoder reads the whole source once; the decoder writes one token at a time with three sublayers: masked self-attention, cross-attention into the encoder's output, and a feed-forward network.
- Two masks keep it honest: the decoder may not read the future, and nobody may read padding.
- Where the LayerNorm sits decides trainability: post-LN (the paper) normalizes after the residual sum and needs warmup; pre-LN normalizes the sublayer's input, needs one final norm per stack, and trains without warmup.
- The paper's recipe is three numbers you can check by hand: embeddings times $\sqrt{d}$, the Noam rate with update $s$ at step $s + 1$, and label smoothing that moves $\varepsilon$ of the target mass onto every id.
- Decoding one token at a time must give the logits of the full forward pass; beam search (`L4.4`) runs on top of that step.

## How to read this chapter

Sections 1 to 3 assemble the architecture from the parts of `L5.2` to `L5.4`, explain the LayerNorm placement and the training recipe, and work the loss and the schedule by hand. Section 4 fixes the interface. Section 5 walks through the solution and explains why each choice is right. Try to predict each step before reading it.

---

## 1. Why now

The seq2seq model of Part 4 (`L4.1` to `L4.3`) reads a sentence one token at a time through a GRU: training cannot run the positions in parallel, and everything the decoder knows about the source passes through attention over states that were themselves built sequentially. The three parts that remove the recurrence are now in place: attention with heads (`L5.3`), masks (`L5.2`), and positions (`L5.4`). This module assembles them into the 2017 architecture, trains it with the paper's recipe, and decodes it with the beam search of `L4.4`. The Part 5 task is three-digit addition, which needs exact attention to two positions per output digit and carries between them: an RNN encoder-decoder of the same size learns it slowly; the transformer reaches 0.98 exact match in a few thousand updates. The same build also exposes the first architectural decision every modern LLM made differently from the paper: where to put the LayerNorm.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x \in \mathbb{R}^{S \times d}$ | source states; $S$ source length | `[B, S, d]` |
| $y \in \mathbb{R}^{T \times d}$ | target states; $T$ target length | `[B, T, d]` |
| $d, H, d_{ff}$ | model width, heads, feed-forward width | `int` |
| $N_e, N_d$ | encoder and decoder layers | `int` |
| $E$ | the shared embedding table | `float32[V, d]` |
| $\text{LN}$ | LayerNorm over the last axis (`L0.4`) | |
| $\text{FF}(x) = W_2\,\text{relu}(W_1 x + b_1) + b_2$ | position-wise feed-forward | `[.., d] -> [.., d]` |
| $V$ | vocabulary size | `int` |
| $\varepsilon$ | label smoothing | `float` in $[0, 1]$ |
| $s$ | update index, 0 for the first update | `int` |
| $w$ | warmup updates | `int` |

### 2.1 The stacks

**Input.** $x_0 = E[\text{src}] \sqrt{d} + \text{PE}$, the same for the target with the same table $E$ (the paper shares one matrix between the two embeddings and the output layer). $E$ starts at $\mathcal{N}(0, 1/d)$, so $E\sqrt d$ has unit scale, the scale of $\text{PE}$.

**Encoder layer** (post-LN): $x \leftarrow \text{LN}_1(x + \text{SelfAttn}(x))$, then $x \leftarrow \text{LN}_2(x + \text{FF}(x))$. Self-attention may read every real source position.

**Decoder layer** (post-LN): $y \leftarrow \text{LN}_1(y + \text{SelfAttn}(y))$ with the causal and target-padding mask, $y \leftarrow \text{LN}_2(y + \text{CrossAttn}(y, m))$ where $m$ is the encoder's output and the mask is the source padding, then $y \leftarrow \text{LN}_3(y + \text{FF}(y))$.

**Output.** $\text{logits} = y E^\top$. With `tie_embeddings` and equal vocabularies, `src_emb`, `tgt_emb`, and `out` are one tensor, listed once in the `state_dict`.

### 2.2 Where the LayerNorm goes

Post-LN computes $\text{LN}(x + F(x))$: every layer's output is normalized, but the gradient that reaches early layers passes through every LN on the residual path, and at initialization the gradients of the last layers are large. With Adam at a fixed learning rate the first updates are too big and training stalls. Xiong et al. show this, and that the fix the paper used, a learning-rate **warmup**, is what makes post-LN trainable.

Pre-LN computes $x + F(\text{LN}(x))$: the residual path is the identity, gradients reach every layer undamped, and no warmup is needed. Because nothing normalizes the stream any more, a pre-LN stack ends with one more LN (`enc_norm`, `dec_norm`). torch's `norm_first=True` is this; GPT-2 (`L6.1`) and every model of Part 7 use it. The LayerNorm demo trains a 6 + 6 layer model at a constant rate $3 \times 10^{-3}$ with no warmup: post-LN keeps about 91% of its first loss after 150 updates, pre-LN about 54%.

### 2.3 Three kinds of attention

| Sublayer | Queries | Keys and values | Mask |
|---|---|---|---|
| encoder self | source | source | source padding `[B, 1, 1, S]` |
| decoder self | target | target | causal `[T, T]` AND target padding `[B, 1, 1, T]` (`L5.2`'s `combine`) |
| decoder cross | target | encoder output | source padding |

The caller passes padding masks only (`src != pad`, `tgt_in != pad`); the model builds these attention masks, so it cannot forget causality.

### 2.4 Training recipe

**Teacher forcing.** A target $y_1 \dots y_T$ is fed as `tgt_in` = (bos, $y_1, \dots, y_{T-1}$) and predicted as $(y_1, \dots, y_T)$, every position at once: the causal mask is what makes this one forward pass equal to $T$ sequential steps.

**Label smoothing.** With $p = \text{softmax}(\text{logits})$ the smoothed target is $q = (1 - \varepsilon)\,\text{onehot}(t) + \varepsilon / V$ and the loss is $-\sum_j q_j \log p_j = (1 - \varepsilon)(-\log p_t) + \varepsilon \cdot \text{mean}_j(-\log p_j)$, with gradient $p - q$. It stops the model from pushing one logit to infinity. `L0.3`'s `cross_entropy(label_smoothing=)` already implements it; `label_smoothed_loss` adds the padding rule: a target equal to the pad id is not a prediction and counts for nothing, not even in the mean.

**Noam schedule** (`M10.4`): $\text{lr}(n) = f\, d^{-1/2} \min(n^{-1/2}, n\, w^{-3/2})$, a linear warmup for $w$ steps, then a decay as $n^{-1/2}$. The paper counts steps from $n = 1$; update $s$ (counting from 0) therefore runs at $n = s + 1$. Evaluating at $n = s$ gives rate 0 to the first update.

**Optimizer** (`M10.3`): Adam with $\beta = (0.9, 0.98)$, $\epsilon = 10^{-9}$, no weight decay.

## 3. Worked example by hand

**Label smoothing**, $V = 3$, logits $(1, 0, 0)$, target 0, $\varepsilon = 0.1$:
$p = (e, 1, 1)/(e + 2) = (0.576117, 0.211942, 0.211942)$, so
$-\log p = (0.551445, 1.551445, 1.551445)$ and

$$\text{loss} = 0.9 \times 0.551445 + 0.1 \times \tfrac{0.551445 + 2 \times 1.551445}{3} = 0.496300 + 0.121811 = 0.618111 .$$

$q = (0.9 + 0.1/3, 0.1/3, 0.1/3) = (0.933333, 0.033333, 0.033333)$, gradient $p - q = (-0.357216, 0.178608, 0.178608)$. A second row whose target is the pad id contributes no loss, no gradient, and does not count in the mean.

**Noam**, $d = 16$, $w = 4$, $f = 1$: $d^{-1/2} = 0.25$.

| update $s$ | $n = s + 1$ | $\min(n^{-1/2}, n w^{-3/2})$ | rate |
|---|---|---|---|
| 0 | 1 | $\min(1, 1/8) = 0.125$ | 0.03125 |
| 3 | 4 | $\min(0.5, 0.5) = 0.5$ | 0.125 (peak) |
| 15 | 16 | $\min(0.25, 2) = 0.25$ | 0.0625 |

## 4. The interface

```python
@dataclass
class TransformerConfig: src_vocab: int; tgt_vocab: int; d_model: int = 512; n_heads: int = 8; d_ff: int = 2048
                         n_enc: int = 6; n_dec: int = 6; dropout: float = 0.1; norm: Literal['post','pre'] = 'post'
                         tie_embeddings: bool = True; max_len: int = 256; ln_eps: float = 1e-5
class Transformer(Module):
    def encode(self, src, src_mask) -> Tensor                                # [B, S, d]
    def decode(self, tgt_in, memory, src_mask, tgt_mask) -> Tensor          # logits [B, T, V]
    def forward(self, src, tgt_in, src_mask, tgt_mask) -> Tensor
    def init_state(self, src_mask) -> DecodeState
    def decode_step(self, y_prev, memory, state) -> tuple[Tensor, DecodeState]   # last position's logits
def label_smoothed_loss(logits, targets, eps, pad_id) -> Tensor
def noam_rate(step, d_model, warmup, factor=1.0) -> float
def fit(model, src, tgt, steps, batch, pad_id, warmup=4000, factor=1.0, smoothing=0.1, lr=None, ...) -> list[float]
def translate(model, src, bos, eos, pad_id, max_len, beam_size=1, length_penalty=1.0) -> list[list[int]]
def save_transformer(model, dir) / load_transformer(dir)
```

Masks passed in are padding masks, bool `[B, S]` and `[B, T]`, True = a real token.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand-worked label smoothing | unit | section 3: 0.618111, the gradient, the ignored pad row | the loss every update uses |
| hand-worked Noam | unit | section 3: 0.03125, 0.125, 0.0625 | the first update learns |
| agreement with torch | golden | torch post-LN and pre-LN with copied weights: logits, smoothed loss, every gradient | both placements match the reference implementation |
| the decoder is causal | property | changing targets after $t$ leaves logits $0..t$ bitwise equal | teacher forcing is honest |
| source padding is never read | property | ids in padded source positions change nothing | batches of different lengths |
| the decoder reads the source | property | a different source changes the logits | cross-attention is wired to the encoder |
| one step at a time equals the full pass | differential | step-by-step decoding equals the full forward pass | beam search scores the right logits |
| embeddings scaled and tied | unit | $E\sqrt d + \text{PE}$; one tied table, listed once | the paper's section 3.4 |
| parameter names follow the norm | unit | pre-LN adds exactly `enc_norm`, `dec_norm`; decoder layers have `cross_attn`, `norm3` | zoo checkpoints (`L6.7`) |
| the paper's Adam | unit | betas (0.9, 0.98), eps 1e-9 | the recipe |
| `fit` follows the schedule and is reproducible | property | exact step count, same seed same losses, a constant `lr` | the post-LN vs pre-LN comparison |
| greedy `translate` with beam 1, eos dropped | differential | beam 1 equals a hand argmax loop; no eos; mode restored; save/load | the `translate` command |
| bad arguments | boundary | unknown norm, non-bool masks, batch mismatch | wiring bugs fail loudly |
| it learns | learning | 600 updates on two-digit reversed addition reach a reference held-out loss | the recipe trains |

A model test suite that rewrites the forward pass in float64 numpy from the `state_dict` (one layer, padding on both sides, both norm orders) is in [`test_transformer.py`](../../../../course/solutions/learner-tests/L5.5/python/tests/l5-5-transformer/test_transformer.py).

## 5. The solution, step by step

### 5.1 From first principles

1. The task is a conditional distribution $p(y_t \mid y_{<t}, x)$. The source is fully known, so it can be read once, in parallel, by a stack of self-attention layers that may look everywhere except padding: the encoder.
2. The target must be generated left to right at inference, but at training time all of $y$ is known. If position $t$ is forbidden from reading positions $> t$, one parallel pass over (bos, $y_1, \dots, y_{T-1}$) computes exactly what $T$ sequential steps would. That is the causal mask, and it is what makes teacher forcing a single forward pass.
3. The decoder still needs the source. A second attention whose queries come from the decoder and whose keys and values come from the encoder output does that: cross-attention, masked by source padding only (there is no "future" in the source).
4. Each sublayer is added to a residual stream so depth does not destroy the signal, and normalized so the scale stays fixed. Where the LN goes is a free choice with consequences (section 2.2): post-LN normalizes the stream itself; pre-LN normalizes only the sublayer's input and so needs one final LN per stack.
5. The embedding table is shared between source, target and output (when vocabularies match), and starts at std $d^{-1/2}$. Multiplying by $\sqrt d$ brings token vectors to unit scale, matching the unit-circle pairs of PE, so neither swamps the other.
6. The masks are derived inside the model from padding masks the caller passes. A caller can then never forget causality, and the same `decode` serves training and decoding.
7. Training is the paper's recipe: label smoothing with pads excluded, Adam (0.9, 0.98, 1e-9), and the Noam rate evaluated at $n = s + 1$ so the first update (worked example: 0.03125) is not zero.
8. Decoding reuses `decode` on the growing prefix and keeps the last row. That is slow ($O(T^2)$ per sequence) but correct by construction; `L8.2` adds a cache. Beam search (`L4.4`) needs only that step function.

### 5.2 The code, annotated

**The layers.** Both placements live in one class; the branch is the whole difference.

```python
class EncoderLayer(Module):
    # ... self_attn, linear1, linear2, norm1, norm2, drop ...

    def ff(self, x: Tensor) -> Tensor:
        return self.linear2(self.drop(F.relu(self.linear1(x))))

    def forward(self, x: Tensor, mask: NDArray) -> Tensor:
        if self.pre:
            # Pre-LN: x + F(LN(x)). The residual path is the identity.
            h = self.norm1(x)
            x = x + self.drop(self.self_attn(h, h, mask))
            return x + self.drop(self.ff(self.norm2(x)))
        # Post-LN: LN(x + F(x)). Every layer's output is normalized.
        x = self.norm1(x + self.drop(self.self_attn(x, x, mask)))
        return self.norm2(x + self.drop(self.ff(x)))
```

The decoder layer adds the cross-attention sublayer between the two:

```python
    def forward(self, y, memory, self_mask, cross_mask):
        if self.pre:
            h = self.norm1(y)
            y = y + self.drop(self.self_attn(h, h, self_mask))
            # Only the query side is normalized; memory is already the encoder's output.
            y = y + self.drop(self.cross_attn(self.norm2(y), memory, cross_mask))
            return y + self.drop(self.ff(self.norm3(y)))
        y = self.norm1(y + self.drop(self.self_attn(y, y, self_mask)))
        # Queries from the decoder, keys and values from the encoder.
        y = self.norm2(y + self.drop(self.cross_attn(y, memory, cross_mask)))
        return self.norm3(y + self.drop(self.ff(y)))
```

**Construction: scaled, tied embeddings and the pre-LN final norms.**

```python
        # Embeddings start at std d^-1/2 so that emb * sqrt(d) has unit scale.
        self.src_emb = Embedding(cfg.src_vocab, d, rng=r)
        self.src_emb.weight.data = normal_init((cfg.src_vocab, d), d**-0.5, r)
        if cfg.tie_embeddings and cfg.src_vocab == cfg.tgt_vocab:
            self.tgt_emb = Embedding(cfg.tgt_vocab, d, rng=r)
            self.tgt_emb.weight = self.src_emb.weight  # one table, listed once
        else:
            # ... a separate target table, same init ...
        self.pos = SinusoidalPE(cfg.max_len, d)
        self.encoder = ModuleList([EncoderLayer(cfg, r) for _ in range(cfg.n_enc)])
        self.decoder = ModuleList([DecoderLayer(cfg, r) for _ in range(cfg.n_dec)])
        if cfg.norm == "pre":
            # Pre-LN leaves the residual stream unnormalized: one last LN per stack.
            self.enc_norm = LayerNorm(d, cfg.ln_eps)
            self.dec_norm = LayerNorm(d, cfg.ln_eps)
        self.out = Linear(d, cfg.tgt_vocab, bias=False, rng=r)
        if cfg.tie_embeddings:
            self.out.weight = self.tgt_emb.weight   # logits = y E^T
```

Assigning the same `Tensor` object (not a copy) is what ties: one gradient accumulates from all three uses, and the `state_dict` lists it once. `enc_norm` and `dec_norm` exist only for pre-LN, so a post-LN checkpoint has exactly the paper's parameter names.

**Embedding, encode, decode.**

```python
    def embed(self, ids, table, offset=0):
        x = table(ids) * math.sqrt(self.cfg.d_model)
        return self.drop(self.pos(x, offset))

    def encode(self, src, src_mask):
        # ... shape and dtype checks ...
        m = _key_mask(src_mask)           # bool [B, S] -> [B, 1, 1, S]
        x = self.embed(src, self.src_emb)
        for layer in self.encoder:
            x = layer(x, m)
        return self.enc_norm(x) if self.cfg.norm == "pre" else x

    def decode(self, tgt_in, memory, src_mask, tgt_mask):
        # ... shape and dtype checks ...
        T = tgt_in.shape[1]
        tm = _key_mask(tgt_mask)
        # Position t may read targets 0..t only: causal AND not padding.
        self_mask = combine(causal_mask(T), tm)
        cross_mask = _key_mask(src_mask)  # cross-attention: source padding only
        y = self.embed(tgt_in, self.tgt_emb)
        for layer in self.decoder:
            y = layer(y, memory, self_mask, cross_mask)
        if self.cfg.norm == "pre":
            y = self.dec_norm(y)
        return self.out(y)
```

`_key_mask` refuses anything but a bool `[B, T]` array. An int mask of 0/1 would be easy to pass by mistake and would broadcast silently with the wrong meaning.

**One decoding step.**

```python
    def init_state(self, src_mask):
        m = np.asarray(src_mask)
        return DecodeState(np.zeros((m.shape[0], 0), dtype=np.int64), m)

    def decode_step(self, y_prev, memory, state):
        y = np.asarray(y_prev, dtype=np.int64).reshape(-1, 1)
        prefix = np.concatenate([state.prefix, y], axis=1)
        # ... batch check ...
        # No cache yet (L8.2 adds one): recompute the whole prefix, keep the last row.
        logits = self.decode(
            prefix, memory, state.src_mask, np.ones(prefix.shape, dtype=bool)
        )
        return logits[:, -1, :], DecodeState(prefix, state.src_mask)
```

Because the decoder is causal, row $t$ of `decode(prefix)` does not depend on anything after $t$, so this equals the full forward pass row for row. The state is immutable (a new `DecodeState` each step), which is what beam search needs when it forks hypotheses.

**The recipe.**

```python
def label_smoothed_loss(logits, targets, eps, pad_id):
    t = np.asarray(targets).astype(np.int64)
    # Padding positions are not predictions: no loss, no gradient, not counted.
    t = np.where(t == pad_id, -100, t)
    return cross_entropy(logits, t, ignore_index=-100, label_smoothing=eps)


def noam_rate(step, d_model, warmup, factor=1.0):
    # Update number `step` (0 first) is the paper's step_num = step + 1.
    return factor * noam(step + 1, d_model, warmup)


def make_optimizer(model):
    # lr is overwritten every step by fit; 0.0 is a placeholder.
    return Adam(list(model.parameters()), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
```

`fit` sets the rate before every update and feeds shifted targets:

```python
    def loss_fn(m, b):
        s, t = b["src"], b["tgt"]
        # Teacher forcing: read bos y_1 .. y_{T-1}, predict y_1 .. y_T.
        logits = m(s, t[:, :-1], s != pad_id, t[:, :-1] != pad_id)
        return label_smoothed_loss(logits, t[:, 1:], smoothing, pad_id)

    # ...
    while step < steps:
        for b in loader:                  # reshuffles each epoch from the seeded stream
            if step >= steps:
                break
            # A constant lr (for the no-warmup demo) or the Noam rate.
            opt.lr = lr if lr is not None else noam_rate(step, model.cfg.d_model, warmup, factor)
            stats = train_step(model, b, loss_fn, opt, clip)
            losses.append(stats["loss"])
            # ...
            step += 1
```

**Translate.** `translate` switches to eval mode (no dropout), encodes each row once, and hands `beam_search` a step function whose state is `(memory, DecodeState)`:

```python
    was = model.training
    model.eval()
    try:
        with no_grad():
            for row in src:
                s = row[None, :]
                mask = s != pad_id
                memory = model.encode(s, mask)

                def step(st, y):
                    mem, ds = st
                    logits, ds2 = model.decode_step(y, mem, ds)
                    return logits.data, (mem, ds2)

                hyps = beam_search(step, (memory, model.init_state(mask)),
                                   bos, eos, beam_size, max_len, length_penalty)
                toks = hyps[0].tokens
                # The answer is the tokens before eos, not eos itself.
                out.append(toks[:-1] if hyps[0].finished else toks)
    finally:
        model.train(was)                  # restore the caller's mode even on error
```

`save_transformer` writes a `config.json` with `tl_arch = "transformer"` and the norm placement, plus `model.safetensors`; `load_transformer` rebuilds the config with dropout 0 and loads the weights. The solution's entry point ([`cli_xfmr.py`](../../../../course/solutions/entry/python/tinyllm/cli_xfmr.py)) wraps these as `train-transformer` (with `--norm`, `--warmup`, `--lr` for the LayerNorm comparison) and `translate --beam`.

Full solution: [`transformer.py`](../../../../course/solutions/python/tinyllm/xfmr/transformer.py).

### 5.3 Why this solution

- **Cost.** One training forward is $O(N_e S^2 d + N_d (T^2 + TS) d + (N_e + N_d)(S + T) d\, d_{ff})$ per sequence, all positions in parallel. Decoding by recomputing the prefix costs $O(T^2)$ decoder passes in total; a KV cache (`L8.2`) makes it $O(T)$. Correctness first, then speed: the cache must reproduce exactly these logits.
- **Masks built inside the model** rather than passed in: one fewer way to train a non-causal decoder that looks perfect in training.
- **One class with a `pre` flag** rather than two classes: the same parameter names (plus `enc_norm`, `dec_norm`) and the same code path, so torch's `norm_first` is matched by flipping one flag.
- **`ignore_index` for pads** rather than masking the loss after the fact: the mean is taken over real targets only, so a batch with short targets is not diluted, and the smoothing mean never includes pad positions.
- **Tradeoff accepted.** Post-LN is the default because it is the paper's model, even though it needs warmup; pre-LN is one flag away, and `L6.1` onward uses it.

### 5.4 Explain it back

1. Why does pre-LN need `enc_norm` and `dec_norm` but post-LN does not?
   <details><summary>Answer</summary>In post-LN the last operation of every layer is a LayerNorm, so the stack's output is already normalized. In pre-LN each layer adds an unnormalized sublayer output to the stream, so after $N$ layers the stream has grown; one final LN brings it back to the scale the output projection (and, for the encoder, cross-attention) expects.</details>
2. Why must `noam_rate` evaluate the schedule at `step + 1`?
   <details><summary>Answer</summary>The Noam formula has a factor $n \cdot w^{-3/2}$ in warmup, which is 0 at $n = 0$. Counting updates from 0, evaluating at $n = s$ would waste the first update and shift the peak one update late.</details>
3. Why does cross-attention use only the source padding mask and no causal mask?
   <details><summary>Answer</summary>The source is entirely known before decoding starts, so there is no future to hide. The only illegal keys are padding positions, which are not real tokens.</details>
4. Why can `decode_step` return `logits[:, -1, :]` of a full `decode` call and still match the full forward pass at every position?
   <details><summary>Answer</summary>The causal mask makes row $t$ a function of rows $0..t$ only. Recomputing the prefix of length $t + 1$ gives exactly row $t$ of the full pass; later tokens cannot change it.</details>
5. What would go wrong if the embeddings were not multiplied by $\sqrt d$?
   <details><summary>Answer</summary>The table starts at std $d^{-1/2}$, so each token vector has norm about 1 while each PE vector has norm $\sqrt{d/2}$. Without the scale, position dominates the input at init and the model has to learn to amplify token identity first.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. post-LN written as pre-LN (normalizing the sublayer's input) | a different model from the paper's; torch disagrees | the `post` branch is `norm(x + sublayer(x))`; see 5.2 |
| 2. pre-LN without the final LayerNorm | the stack's output is unnormalized; logits blow up with depth | `enc_norm` / `dec_norm` created and applied when `norm == "pre"` |
| 3. no causal mask in the decoder | training loss near zero, generation garbage | `decode` always builds `combine(causal_mask(T), tm)` itself |
| 4. cross-attention reading the decoder instead of the encoder | trains as a language model of the targets, never learns the task | `cross_attn(y, memory, ...)`: keys and values are `memory` |
| 5. embeddings not multiplied by $\sqrt d$ | positions swamp the tokens at init | `embed` multiplies by `math.sqrt(d_model)` before adding PE |
| 6. source padding readable in cross-attention | outputs depend on how long the batch's other sentences are | `cross_mask = _key_mask(src_mask)` in every decoder layer |
| 7. label smoothing counting pad targets | short targets in a batch pull every logit toward the pad id | pads mapped to `ignore_index = -100` before `cross_entropy` |
| 8. Noam at $n = s$ | the first update has rate 0, the peak is one late | `noam(step + 1, ...)` |
| the output layer not tied | twice the parameters; the zoo's keys differ | `self.out.weight = self.tgt_emb.weight` |
| `decode_step` returning the first position | beam search scores the wrong token | returns `logits[:, -1, :]` |
| `translate` keeping eos | exact match is 0 | `toks[:-1] if hyps[0].finished` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L5.2` | `causal_mask` and `combine` build the decoder mask |
| Back | `L5.3` | every attention sublayer |
| Back | `L5.4` | `SinusoidalPE` on source and target |
| Back | `L0.1` | `Tensor` and `no_grad` for decoding |
| Back | `L0.2` | `relu`, the ops of every layer |
| Back | `L0.3` | `cross_entropy` with `label_smoothing` |
| Back | `L0.4` | `Linear`, `Embedding`, `LayerNorm`, `ModuleList` |
| Back | `L0.5` | `DataLoader` and `train_step` inside `fit` |
| Back | `L0.6` | safetensors for the model directory |
| Back | `M10.3` | `Adam` |
| Back | `M10.4` | `noam` |
| Back | `M07.3` | `normal_init` for the $\mathcal{N}(0, 1/d)$ table |
| Back | `M06.3` | `PCG32` init and shuffle streams |
| Back | `L4.4` | `beam_search` over `decode_step` |
| Forward | `L6.1` | GPT keeps only the decoder, in pre-LN |
| Forward | `L6.2` | BERT keeps only the encoder, in post-LN |
| Forward | `L6.7` | the zoo loads `tl_arch = transformer` and reports exact match with beam search |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Transformer` | torch `nn.Transformer`, fairseq `TransformerModel` | fused attention, `norm_first`, incremental state caching | `torch/nn/modules/transformer.py`; `fairseq/models/transformer/` |
| post-LN vs pre-LN | DeepNorm, Sandwich-LN, RMSNorm pre-norm | train 1000-layer post-LN; modern LLMs use pre-RMSNorm (`L7.1`) | Wang et al., "DeepNet" (2022) |
| `noam_rate` | linear warmup + cosine, WSD | the schedules LLM training uses (`M10.4`, `C1`) | HF `get_cosine_schedule_with_warmup` |
| `decode_step` recomputing the prefix | the KV cache | $O(T)$ instead of $O(T^2)$ work per token | `L8.2`; HF `past_key_values` |
