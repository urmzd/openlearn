# Encoder-decoder Transformer: post-LN and pre-LN, Noam, label smoothing

The seq2seq model of the previous chapter, from [Encoder-decoder with teacher forcing](../05-attention-origins/01-encoder-decoder-teacher-forcing.md) through [Luong attention](../05-attention-origins/03-luong-attention-input-feeding.md), reads a sentence one token at a time through a GRU. Training cannot run the positions in parallel, and everything the decoder knows about the source passes through attention over states that were themselves built sequentially. The three parts that remove the recurrence are now in place: [attention with heads](03-multi-head-attention.md), [masks](02-masks.md), and [positions](04-positional-encodings.md).

This section assembles them into the 2017 architecture, trains it with the paper's recipe, and decodes it with the [beam search](../05-attention-origins/04-beam-search.md) you already built. The task is three-digit addition, which needs exact attention to two positions per output digit and carries between them. An RNN encoder-decoder of the same size learns it slowly; the transformer reaches 0.98 exact match on unseen sums in a few thousand updates. The same build also exposes the first architectural decision every modern LLM made differently from the paper: where to put the LayerNorm. You will see post-LN fail without warmup while pre-LN trains.

Besides the three sections above, this one uses most of the foundations chapter: the [`Tensor`](../01-foundations/02-tensor-and-broadcasting-backward.md), the [op library](../01-foundations/03-op-library-and-gradcheck.md), `cross_entropy` from [Losses with a fused backward](../01-foundations/04-losses-with-fused-backward.md), the layers of [Module system and basic layers](../01-foundations/05-module-system-and-layers.md), `DataLoader` and `train_step` from [Training loop and the autograd bigram](../01-foundations/06-training-loop-and-the-autograd-bigram.md), and safetensors from [Safetensors for every dtype, atomic checkpoints, and the token stream](../01-foundations/07-safetensors-checkpoints-and-token-streams.md). From *Mathematics for Computing* it uses `Adam` from [Adam and AdamW](../../mathematics-for-computing/11-optimization/03-adam-and-adamw.md), `noam` from [Learning-rate schedules and gradient clipping](../../mathematics-for-computing/11-optimization/04-schedules-and-gradient-clipping.md), `normal_init` from [Variance propagation and initialization](../../mathematics-for-computing/08-probability-statistics/04-variance-propagation-and-initialization.md), and `PCG32` from [Modular arithmetic, hashing, and PCG32](../../mathematics-for-computing/07-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md). The full solution is [`python/tinyllm/xfmr/transformer.py`](../../../../course/solutions/python/tinyllm/xfmr/transformer.py), its contract is [`course/contracts/py/tinyllm/xfmr/transformer.pyi`](../../../../course/contracts/py/tinyllm/xfmr/transformer.pyi), and the model oracle tests are in [`test_transformer.py`](../../../../course/solutions/learner-tests/L5.5/python/tests/l5-5-transformer/test_transformer.py).

> **Learning objectives**
> - Assemble encoder and decoder stacks from attention, masks, positions, and a feed-forward network, with all three kinds of attention wired correctly.
> - Explain why post-LN needs warmup and pre-LN does not, and why pre-LN needs one final norm per stack.
> - Compute label-smoothed loss and the Noam rate by hand, including the pad rule and the $n = s + 1$ step count.
> - Show that step-by-step decoding reproduces the full forward pass, and run beam search on top of it.

## The notation

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x \in \mathbb{R}^{S \times d}$ | source states; $S$ source length | `[B, S, d]` |
| $y \in \mathbb{R}^{T \times d}$ | target states; $T$ target length | `[B, T, d]` |
| $d, H, d_{ff}$ | model width, heads, feed-forward width | `int` |
| $N_e, N_d$ | encoder and decoder layers | `int` |
| $E$ | the shared embedding table | `float32[V, d]` |
| $\text{LN}$ | LayerNorm over the last axis | |
| $\text{FF}(x) = W_2\,\text{relu}(W_1 x + b_1) + b_2$ | position-wise feed-forward | `[.., d] -> [.., d]` |
| $V$ | vocabulary size | `int` |
| $\varepsilon$ | label smoothing | `float` in $[0, 1]$ |
| $s$ | update index, 0 for the first update | `int` |
| $w$ | warmup updates | `int` |

## The idea, from first principles

The task is a conditional distribution $p(y_t \mid y_{<t}, x)$. The source is fully known, so it can be read once, in parallel, by a stack of self-attention layers that may look everywhere except padding. That stack is the encoder.

The target must be generated left to right at inference, but at training time all of $y$ is known. If position $t$ is forbidden from reading positions $> t$, one parallel pass over (bos, $y_1, \dots, y_{T-1}$) computes exactly what $T$ sequential steps would. That is the causal mask, and it is what makes teacher forcing a single forward pass.

The decoder still needs the source. A second attention whose queries come from the decoder and whose keys and values come from the encoder output does that: cross-attention, masked by source padding only, since there is no "future" in the source. Each sublayer is added to a residual stream so depth does not destroy the signal, and normalized so the scale stays fixed. Where the normalization goes turns out to be a free choice with consequences.

## The stacks

**Input.** $x_0 = E[\text{src}] \sqrt{d} + \text{PE}$, and the same for the target with the same table $E$ (the paper shares one matrix between the two embeddings and the output layer). $E$ starts at $\mathcal{N}(0, 1/d)$, so $E\sqrt d$ has unit scale, the scale of the unit-circle pairs of $\text{PE}$, and neither swamps the other.

**Encoder layer** (post-LN): $x \leftarrow \text{LN}_1(x + \text{SelfAttn}(x))$, then $x \leftarrow \text{LN}_2(x + \text{FF}(x))$. Self-attention may read every real source position.

**Decoder layer** (post-LN): $y \leftarrow \text{LN}_1(y + \text{SelfAttn}(y))$ with the causal and target-padding mask, then $y \leftarrow \text{LN}_2(y + \text{CrossAttn}(y, m))$ where $m$ is the encoder's output and the mask is the source padding, then $y \leftarrow \text{LN}_3(y + \text{FF}(y))$.

**Output.** $\text{logits} = y E^\top$. With `tie_embeddings` and equal vocabularies, `src_emb`, `tgt_emb`, and `out` are one tensor, listed once in the `state_dict`.

## Where the LayerNorm goes

Post-LN computes $\text{LN}(x + F(x))$. Every layer's output is normalized, but the gradient that reaches early layers passes through every LN on the residual path, and at initialization the gradients of the last layers are large. With Adam at a fixed learning rate the first updates are too big and training stalls. Xiong et al. show this, and that the fix the paper used, a learning-rate **warmup**, is what makes post-LN trainable.

Pre-LN computes $x + F(\text{LN}(x))$. The residual path is the identity, gradients reach every layer undamped, and no warmup is needed. Because nothing normalizes the stream any more, a pre-LN stack ends with one more LN (`enc_norm`, `dec_norm`). torch's `norm_first=True` is this, and [GPT](../07-objectives/01-gpt-decoder-only.md) and every model of the modern block use it.

The difference is easy to see. The LayerNorm demo trains a 6 + 6 layer model at a constant rate $3 \times 10^{-3}$ with no warmup: post-LN keeps about 91% of its first loss after 150 updates, pre-LN about 54%.

## Three kinds of attention

The model uses multi-head attention in three roles, each with its own sources and mask:

| Sublayer | Queries | Keys and values | Mask |
|---|---|---|---|
| encoder self | source | source | source padding `[B, 1, 1, S]` |
| decoder self | target | target | causal `[T, T]` AND target padding `[B, 1, 1, T]` (via `combine`) |
| decoder cross | target | encoder output | source padding |

The caller passes padding masks only (`src != pad`, `tgt_in != pad`). The model builds these attention masks itself, so it cannot forget causality.

## The training recipe

**Teacher forcing.** A target $y_1 \dots y_T$ is fed as `tgt_in` = (bos, $y_1, \dots, y_{T-1}$) and predicted as $(y_1, \dots, y_T)$, every position at once. The causal mask is what makes this one forward pass equal to $T$ sequential steps.

**Label smoothing.** With $p = \text{softmax}(\text{logits})$ the smoothed target is $q = (1 - \varepsilon)\,\text{onehot}(t) + \varepsilon / V$ and the loss is $-\sum_j q_j \log p_j = (1 - \varepsilon)(-\log p_t) + \varepsilon \cdot \text{mean}_j(-\log p_j)$, with gradient $p - q$. It stops the model from pushing one logit to infinity. The foundations' `cross_entropy(label_smoothing=)` already implements it; `label_smoothed_loss` adds the padding rule: a target equal to the pad id is not a prediction and counts for nothing, not even in the mean.

**Noam schedule.** $\text{lr}(n) = f\, d^{-1/2} \min(n^{-1/2}, n\, w^{-3/2})$: a linear warmup for $w$ steps, then a decay as $n^{-1/2}$. The paper counts steps from $n = 1$, so update $s$ (counting from 0) runs at $n = s + 1$. Evaluating at $n = s$ gives rate 0 to the first update.

**Optimizer.** Adam with $\beta = (0.9, 0.98)$, $\epsilon = 10^{-9}$, and no weight decay.

## Working it by hand

Two pieces of the recipe are easy to get subtly wrong, so work both. First, **label smoothing** with $V = 3$, logits $(1, 0, 0)$, target 0, and $\varepsilon = 0.1$. The softmax is $p = (e, 1, 1)/(e + 2) = (0.576117, 0.211942, 0.211942)$, so $-\log p = (0.551445, 1.551445, 1.551445)$ and

$$\text{loss} = 0.9 \times 0.551445 + 0.1 \times \tfrac{0.551445 + 2 \times 1.551445}{3} = 0.496300 + 0.121811 = 0.618111 .$$

The smoothed target is $q = (0.9 + 0.1/3, 0.1/3, 0.1/3) = (0.933333, 0.033333, 0.033333)$, and the gradient is $p - q = (-0.357216, 0.178608, 0.178608)$. A second row whose target is the pad id contributes no loss, no gradient, and does not count in the mean.

Second, **Noam** with $d = 16$, $w = 4$, $f = 1$, so $d^{-1/2} = 0.25$:

| update $s$ | $n = s + 1$ | $\min(n^{-1/2}, n w^{-3/2})$ | rate |
|---|---|---|---|
| 0 | 1 | $\min(1, 1/8) = 0.125$ | 0.03125 |
| 3 | 4 | $\min(0.5, 0.5) = 0.5$ | 0.125 (peak) |
| 15 | 16 | $\min(0.25, 2) = 0.25$ | 0.0625 |

The first update learns at 0.03125, and the peak falls exactly at the last warmup update.

## Designing the API

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

The masks passed in are padding masks, bool `[B, S]` and `[B, T]`, True = a real token. Decoding reuses `decode` on the growing prefix and keeps the last row. That is slow ($O(T^2)$ per sequence) but correct by construction; the [KV cache](../09-inference/02-kv-cache-and-generate.md) adds the speed later. Beam search needs only that step function.

## Building it step by step

### The layers

Both placements live in one class; the branch is the whole difference.

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

### Construction: scaled, tied embeddings and the final norms

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

### Embedding, encode, decode

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

### One decoding step

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

### The recipe

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

### Translating

`translate` switches to eval mode (no dropout), encodes each row once, and hands `beam_search` a step function whose state is `(memory, DecodeState)`:

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

`save_transformer` writes a `config.json` with `tl_arch = "transformer"` and the norm placement, plus `model.safetensors`; `load_transformer` rebuilds the config with dropout 0 and loads the weights. The solution's entry point ([`cli_xfmr.py`](../../../../course/solutions/entry/python/tinyllm/cli_xfmr.py)) wraps these as `train-transformer` (with `--norm`, `--warmup`, `--lr` for the LayerNorm comparison) and `translate --beam`. The full solution is [`transformer.py`](../../../../course/solutions/python/tinyllm/xfmr/transformer.py).

### Why this design

One training forward costs $O(N_e S^2 d + N_d (T^2 + TS) d + (N_e + N_d)(S + T) d\, d_{ff})$ per sequence, all positions in parallel. Decoding by recomputing the prefix costs $O(T^2)$ decoder passes in total; a KV cache makes it $O(T)$. Correctness comes first, then speed: the cache must reproduce exactly these logits.

The masks are built inside the model rather than passed in, which removes one way to train a non-causal decoder that looks perfect in training. One class with a `pre` flag, rather than two classes, keeps the same parameter names (plus `enc_norm`, `dec_norm`) and the same code path, so torch's `norm_first` is matched by flipping one flag. Using `ignore_index` for pads, rather than masking the loss after the fact, takes the mean over real targets only, so a batch with short targets is not diluted and the smoothing mean never includes pad positions.

One tradeoff is accepted: post-LN is the default because it is the paper's model, even though it needs warmup. Pre-LN is one flag away, and everything from GPT onward uses it.

## What can go wrong

A correct implementation must reproduce both hand examples: the label-smoothed loss of 0.618111 with its gradient and the ignored pad row, and the Noam rates 0.03125, 0.125, and 0.0625, so the first update actually learns. Against torch, post-LN and pre-LN models with copied weights must match on logits, smoothed loss, and every gradient.

The properties follow from the design. Changing targets after $t$ must leave logits $0..t$ bitwise equal (the decoder is causal, so teacher forcing is honest). Ids in padded source positions must change nothing, so batches of different lengths behave. A different source must change the logits, proving cross-attention is wired to the encoder. Step-by-step decoding must equal the full forward pass, so beam search scores the right logits.

The structural checks: embeddings are $E\sqrt d + \text{PE}$ with one tied table listed once (the paper's section 3.4); pre-LN adds exactly `enc_norm` and `dec_norm`, and decoder layers have `cross_attn` and `norm3`, which the [model zoo](../07-objectives/07-lm-evaluation-harness-and-the-model-zoo.md) checkpoints depend on; and Adam uses betas (0.9, 0.98) and eps 1e-9. `fit` must take the exact step count, give the same losses for the same seed, and honour a constant `lr` (needed for the post-LN vs pre-LN comparison). Greedy `translate` with beam 1 must equal a hand argmax loop, drop eos, restore the caller's mode, and survive save/load. An unknown norm, non-bool masks, and a batch mismatch must raise. Finally, it must learn: 600 updates on two-digit reversed addition reach a reference held-out loss.

The model suite [`test_transformer.py`](../../../../course/solutions/learner-tests/L5.5/python/tests/l5-5-transformer/test_transformer.py) rewrites the forward pass in float64 numpy from the `state_dict` (one layer, padding on both sides, both norm orders) and compares against it.

The pitfalls, several of which produce a model that trains happily and is still wrong:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| post-LN written as pre-LN (normalizing the sublayer's input) | a different model from the paper's; torch disagrees | the `post` branch is `norm(x + sublayer(x))` |
| pre-LN without the final LayerNorm | the stack's output is unnormalized; logits blow up with depth | `enc_norm` / `dec_norm` created and applied when `norm == "pre"` |
| no causal mask in the decoder | training loss near zero, generation garbage | `decode` always builds `combine(causal_mask(T), tm)` itself |
| cross-attention reading the decoder instead of the encoder | trains as a language model of the targets, never learns the task | `cross_attn(y, memory, ...)`: keys and values are `memory` |
| embeddings not multiplied by $\sqrt d$ | positions swamp the tokens at init | `embed` multiplies by `math.sqrt(d_model)` before adding PE |
| source padding readable in cross-attention | outputs depend on how long the batch's other sentences are | `cross_mask = _key_mask(src_mask)` in every decoder layer |
| label smoothing counting pad targets | short targets in a batch pull every logit toward the pad id | pads mapped to `ignore_index = -100` before `cross_entropy` |
| Noam at $n = s$ | the first update has rate 0, the peak is one late | `noam(step + 1, ...)` |
| the output layer not tied | twice the parameters; the zoo's keys differ | `self.out.weight = self.tgt_emb.weight` |
| `decode_step` returning the first position | beam search scores the wrong token | returns `logits[:, -1, :]` |
| `translate` keeping eos | exact match is 0 | `toks[:-1] if hyps[0].finished` |

> **Warning:** a missing causal mask is the most expensive bug on this list, because it looks like success. Training loss drops to nearly zero, and only generation reveals that the model learned to copy.

## Where this leads

The next chapter splits this model in two. [GPT](../07-objectives/01-gpt-decoder-only.md) keeps only the decoder, in pre-LN; [BERT](../07-objectives/02-bert-masked-lm.md) keeps only the encoder, in post-LN. The [model zoo](../07-objectives/07-lm-evaluation-harness-and-the-model-zoo.md) loads `tl_arch = transformer` checkpoints and reports exact match with beam search on the addition task.

The production equivalents show where each piece goes next:

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Transformer` | torch `nn.Transformer`, fairseq `TransformerModel` | fused attention, `norm_first`, incremental state caching | `torch/nn/modules/transformer.py`; `fairseq/models/transformer/` |
| post-LN vs pre-LN | DeepNorm, Sandwich-LN, RMSNorm pre-norm | train 1000-layer post-LN; modern LLMs use pre-RMSNorm (see [Pre-LN and RMSNorm](../08-modern-block/01-pre-ln-rmsnorm.md)) | Wang et al., "DeepNet" (2022) |
| `noam_rate` | linear warmup + cosine, WSD | the schedules LLM training uses (see [Learning-rate schedules](../../mathematics-for-computing/11-optimization/04-schedules-and-gradient-clipping.md) and the capstone) | HF `get_cosine_schedule_with_warmup` |
| `decode_step` recomputing the prefix | the KV cache | $O(T)$ instead of $O(T^2)$ work per token | [KV cache and generate](../09-inference/02-kv-cache-and-generate.md); HF `past_key_values` |

### Further reading

- Vaswani et al., "Attention Is All You Need" (2017), sections 3 and 5.
- Xiong et al., "On Layer Normalization in the Transformer Architecture" (ICML 2020).
- Szegedy et al., "Rethinking the Inception Architecture" (2016), section 7 (label smoothing).
- Rush, *The Annotated Transformer*.

## Key terms

- **encoder**: a stack of self-attention and feed-forward layers that reads the whole source once.
- **decoder**: a stack with masked self-attention, cross-attention into the encoder output, and a feed-forward network.
- **cross-attention**: attention with queries from the decoder and keys and values from the encoder output.
- **post-LN**: $\text{LN}(x + F(x))$, the paper's placement; needs warmup.
- **pre-LN**: $x + F(\text{LN}(x))$ with one final LN per stack; trains without warmup.
- **label smoothing**: training against $(1 - \varepsilon)\,\text{onehot}(t) + \varepsilon/V$ instead of a one-hot target.
- **Noam schedule**: $f\, d^{-1/2} \min(n^{-1/2}, n\, w^{-3/2})$, linear warmup then inverse-square-root decay.

## Summary

- The encoder reads the whole source once; the decoder writes one token at a time with three sublayers: masked self-attention, cross-attention into the encoder's output, and a feed-forward network.
- Two masks keep it honest: the decoder may not read the future, and nobody may read padding.
- Where the LayerNorm sits decides trainability: post-LN (the paper) normalizes after the residual sum and needs warmup; pre-LN normalizes the sublayer's input, needs one final norm per stack, and trains without warmup.
- The paper's recipe is three numbers you can check by hand: embeddings times $\sqrt{d}$, the Noam rate with update $s$ at step $s + 1$, and label smoothing that moves $\varepsilon$ of the target mass onto every id.
- Decoding one token at a time must give the logits of the full forward pass; beam search runs on top of that step.

## Review questions

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
