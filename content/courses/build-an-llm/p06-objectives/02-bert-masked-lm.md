# BERT: the bidirectional encoder and masked language modeling

The [GPT of the previous section](01-gpt-decoder-only.md) reads left to right. That is exactly what generation needs and exactly what classification does not: to decide whether "not bad at all" is positive, every word should see every other word. A bidirectional encoder, though, cannot be trained by next-token prediction, because each position could simply read the answer. BERT's fix is to hide some tokens and ask for them back.

The system needs this encoder twice. [ELECTRA](03-electra-replaced-token-detection.md) trains a small BERT as its generator and a BERT-shaped discriminator, and the classification heads of [fine-tuning heads and the linear-head export](05-fine-tuning-heads-and-the-linear-head-export.md), including the usage-policy head the gateway evaluates, sit on top of it. Its tokenizer is the [WordPiece tokenizer](../p01-tokenizers/03-wordpiece.md) of Part 1.

This section builds on [multi-head attention](../p05-transformer-2017/03-multi-head-attention.md), [learned positional encodings](../p05-transformer-2017/04-positional-encodings.md), the encoder of [the encoder-decoder Transformer](../p05-transformer-2017/05-encoder-decoder-transformer.md), and the Part 0 foundations ([`Tensor`](../p00-foundations/01-tensor-and-broadcasting-backward.md), [ops](../p00-foundations/02-op-library-and-gradcheck.md), [`cross_entropy`](../p00-foundations/03-losses-with-fused-backward.md), [layers](../p00-foundations/04-module-system-and-layers.md), [safetensors](../p00-foundations/06-safetensors-checkpoints-and-token-streams.md)). From *Mathematics for Computing* it uses [categorical sampling](../../mathematics-for-computing/07-probability-statistics/01-categorical-sampling.md) (`sample_categorical`), [PCG32](../../mathematics-for-computing/06-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md), and [`normal_init`](../../mathematics-for-computing/07-probability-statistics/03-variance-propagation-and-initialization.md). The reference implementation is [`python/tinyllm/obj/bert.py`](../../../../course/solutions/python/tinyllm/obj/bert.py) (`mlm_mask`, `special_ids`, `mlm_batch`, `BertConfig`, `BertLayer`, `BertEncoder`, `BertForMLM`, `hf_bert_encoder_sd`, `load_hf_bert`, `save_bert`, `load_bert`), with the contract in [`course/contracts/py/tinyllm/obj/bert.pyi`](../../../../course/contracts/py/tinyllm/obj/bert.pyi).

> **Learning objectives**
> - Explain why a bidirectional encoder needs a corruption objective rather than next-token prediction.
> - Apply the 15% pick and 80/10/10 replacement rule by hand, draw for draw, in the fixed op order.
> - Derive the exact shares a picked position shows, including the $1/V$ corrections.
> - Build BERT's post-LN encoder with exact GELU, three summed embeddings, and a tied MLM head with its own bias.
> - Load a Hugging Face BERT checkpoint into either the MLM model or a bare encoder.

## The encoder reads both ways

Here is the notation for this section.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x_t$ | input id at position $t$ | `int` |
| $s_t \in \{0, 1\}$ | special position ([CLS], [SEP], [PAD], ...): never picked | `bool` |
| $p$ | picking probability, 0.15 | `float` |
| $u$ | one uniform draw from the PCG32 | `float` in $[0, 1)$ |
| $V$ | vocabulary size | `int` |
| $y_t$ | label: the original id at picked positions, $-100$ elsewhere | `int` |
| $\text{tt}_t$ | token type (segment A = 0, B = 1) | `int` |
| $W_{word}, W_{pos}, W_{type}$ | the three embedding tables | `float32[V, d]`, `[L, d]`, `[2, d]` |
| $\Phi$ | the standard normal CDF; exact GELU is $u\,\Phi(u)$ | |

The encoder sums three embeddings and normalizes them:

$$h_0 = \text{LN}(W_{word}[x] + W_{pos}[0..T-1] + W_{type}[\text{tt}]),$$

then applies $L$ post-LN layers: $h \leftarrow \text{LN}(h + \text{Attn}(h))$, $h \leftarrow \text{LN}(h + W_2\,\text{GELU}(W_1 h))$. The activation is the exact GELU $u\,\Phi(u)$ (Hugging Face's `"gelu"`), and LayerNorm uses $\epsilon = 10^{-12}$. The attention mask covers padding only, with shape `[B, 1, 1, T]`: every real position reads every other real position, to its left and to its right. Changing the last word changes the first position's output, which a GPT can never do.

## Masking 15%, and the 80/10/10 rule

Because every position can see every other, the objective has to hide information. Masking visits positions in C order (row by row):

1. A special position draws nothing; its label is $-100$.
2. Otherwise draw $u$. If $u \ge p$ the position is not picked; its label is $-100$.
3. Otherwise set the label $y_t = x_t$ (the original) and draw the action $a$ = `sample_categorical([0.8, 0.1, 0.1], u')` by inverse CDF.
4. If $a = 0$, show [MASK]; if $a = 1$, show `rng.below(V)`, a uniform random id; if $a = 2$, show $x_t$ itself.

Why not always show [MASK]? Because [MASK] never appears when the encoder is fine-tuned or used, so a model trained only on [MASK] positions would learn that real tokens need no thought. The random 10% teaches it that any visible token may be wrong, and the kept 10% teaches it that a visible token may be right. The label is always the original token, also at kept positions, because those positions are predictions too. Draws happen only where the steps say, so the same seed gives the same batch in Python, Rust, and Go.

One subtlety affects how you test the rates. The random id is uniform over all $V$ ids, so it can be [MASK] or the original token by chance. The exact shares of what a picked position **shows** are therefore $0.8 + 0.1/V$ for [MASK], $0.1 (V - 2)/V$ for another token, and $0.1 + 0.1/V$ for itself. A statistical check of the rates must use these, not the nominal 80/10/10.

## The MLM head and loss

The corrupted batch goes through the encoder, and a small head turns each position back into a distribution over the vocabulary:

$$z_t = \text{LN}(\text{GELU}(W_T h_t + b_T))\,W_{word}^\top + b_{dec}, \qquad \mathcal{L} = \frac{1}{|\{t : y_t \ne -100\}|} \sum_{y_t \ne -100} -\log \operatorname{softmax}(z_t)_{y_t}.$$

The decoder weight is the word table (tied), and only its bias $b_{dec}$ is new. Positions that were not picked contribute no loss and no gradient, through `cross_entropy`'s `ignore_index`.

## Hugging Face names

Loading BERT is easier than loading GPT-2. Hugging Face's BERT uses `nn.Linear`, so there are no transposes (unlike GPT-2's `Conv1D`). The keys are `embeddings.{word,position,token_type}_embeddings`, `embeddings.LayerNorm`, `encoder.layer.i.attention.self.{query,key,value}`, `attention.output.dense`, `attention.output.LayerNorm`, `intermediate.dense`, `output.dense`, `output.LayerNorm`, and for the head `cls.predictions.transform.dense`, `transform.LayerNorm`, and `cls.predictions.bias`. Hugging Face's attention mask adds the most negative float instead of $-\infty$; a padded key gets weight 0 either way.

## Working one batch by hand

The op order is easiest to internalize by running it. Take ids `[CLS] the cat sat [SEP] [PAD]` = $(2, 5, 6, 7, 3, 0)$, [MASK] = 4, $V = 40$, $p = 0.15$, and a generator that will return the uniforms $0.05, 0.50, 0.90, 0.10, 0.85$ and then `below(40)` = 9. The table traces each position.

| position | token | special? | draws | picked? | action | shown | label |
|---|---|---|---|---|---|---|---|
| 0 | [CLS] | yes | none | | | 2 | -100 |
| 1 | the | no | $u = 0.05 < 0.15$; $a$: $0.50 < 0.8$ | yes | [MASK] | 4 | 5 |
| 2 | cat | no | $u = 0.90$ | no | | 6 | -100 |
| 3 | sat | no | $u = 0.10$; $a$: $0.8 \le 0.85 < 0.9$; below = 9 | yes | random | 9 | 7 |
| 4 | [SEP] | yes | none | | | 3 | -100 |
| 5 | [PAD] | yes | none | | | 0 | -100 |

The result is inputs $(2, 4, 6, 9, 3, 0)$ and labels $(-100, 5, -100, 7, -100, -100)$, using exactly five uniforms and one `below`.

## Designing the interface

```python
def mlm_mask(ids, special_mask, mask_id: int, vocab: int, p: float, rng) -> tuple[NDArray, NDArray]  # inputs, labels
def special_ids(tok: WordPieceTokenizer) -> list[int]
def mlm_batch(tok, texts, max_len: int, p: float, rng) -> dict[str, NDArray]   # input_ids, token_type_ids, attention_mask, labels
@dataclass
class BertConfig: vocab: int; max_len: int = 512; d_model: int = 768; n_heads: int = 12; n_layers: int = 12
                  d_ff: int = 3072; type_vocab: int = 2; dropout: float = 0.1; ln_eps: float = 1e-12
class BertEncoder(Module): def forward(self, ids, token_type_ids=None, attn_mask=None) -> Tensor
class BertForMLM(Module):  def forward(self, ids, token_type_ids=None, attn_mask=None, labels=None) -> (logits, loss)
def load_hf_bert(model, sd) -> None;  def save_bert(model, dir); def load_bert(dir) -> BertForMLM
```

Masking is a pure function of ids, a special mask, and a generator; `mlm_batch` wraps it to go straight from text. The encoder and the MLM model are separate classes so that downstream tasks can start from a bare encoder.

## Building it step by step

The design follows from a chain of requirements.

1. Classification wants every token to see every other, so the encoder attends both ways and masks only padding. Then next-token prediction is impossible (each position can read the answer), so the objective must hide information: corrupt some inputs and predict the originals.
2. Only corrupted positions are informative, so only they are labelled; all others get $-100$ and drop out of the loss. Special positions ([CLS], [SEP], [PAD]) are never picked, because predicting them teaches nothing about language.
3. If the corruption were always [MASK], the encoder could ignore every visible token, and [MASK] never appears downstream. Mixing in random tokens (10%) and the token itself (10%) forces the model to judge every visible token. The label is always the **original**, including at kept positions.
4. Reproducibility across languages demands a fixed draw order: positions in C order, specials draw nothing, one uniform to pick, and only for picked positions one uniform for the action and, for a random action, one `below(V)`. In the worked example that is exactly five uniforms and one `below`.
5. The architecture is the 2017 encoder in post-LN with BERT's details: exact GELU, three summed embeddings (word, position, type) followed by a LayerNorm, and $\epsilon = 10^{-12}$.
6. The MLM head reuses the word table as its output matrix, so only a dense transform, a LayerNorm, and a bias are new. Hugging Face uses `nn.Linear`, so the loader is a pure rename with no transposes.

### Masking

One loop over the flattened batch puts every draw exactly where the op order says:

```python
MASK, RANDOM, KEEP = 0, 1, 2
SPLIT = np.array([0.8, 0.1, 0.1])  # what a picked position shows: [MASK], random, itself

def mlm_mask(ids, special_mask, mask_id, vocab, p, rng):
    x = np.asarray(ids)
    sp = np.asarray(special_mask, dtype=bool)
    # ... dtype, shape, p and mask_id checks ...
    inputs = x.astype(np.int64).copy()          # never modify the caller's ids
    labels = np.full(x.shape, -100, dtype=np.int64)
    # reshape(-1) of a fresh C-contiguous array is a view: writing flat_* writes inputs/labels.
    flat_in, flat_lab, flat_sp = inputs.reshape(-1), labels.reshape(-1), sp.reshape(-1)
    for i in range(flat_in.size):  # C order; a special position draws nothing
        if flat_sp[i]:
            continue
        if rng.uniform() >= p:     # draw 1: picked with probability p
            continue
        flat_lab[i] = flat_in[i]  # the loss asks for the ORIGINAL token here
        action = sample_categorical(SPLIT, rng.uniform())  # draw 2 (inverse CDF)
        if action == MASK:
            flat_in[i] = mask_id
        elif action == RANDOM:
            flat_in[i] = rng.below(vocab)      # draw 3, only for RANDOM
        # KEEP: the input shows the token itself, but it is still predicted.
    return inputs, labels
```

The label is written before the input is overwritten, which is what makes it the original rather than what was shown.

### Batches from text

`mlm_batch` tokenizes, truncates while keeping `[SEP]` last, pads, and marks as special every special id and every padded position:

```python
    special = np.isin(ids, special_ids(tok)) | ~attn   # specials and padding never picked
    vocab = max(tok.vocab.values()) + 1
    inputs, labels = mlm_mask(ids, special, mask_id, vocab, p, rng)
```

### The layer and the encoder

```python
class BertLayer(Module):
    # ... attn, attn_norm, ff1, ff2, ff_norm; every Linear at N(0, 0.02^2) ...
    def forward(self, x: Tensor, mask: NDArray) -> Tensor:
        # Post-LN, as the 2017 transformer (norm="post"), exact GELU.
        x = self.attn_norm(x + self.drop(self.attn(x, x, mask)))
        return self.ff_norm(x + self.drop(self.ff2(F.gelu(self.ff1(x)))))
```

```python
    def forward(self, ids, token_type_ids=None, attn_mask=None):
        # ... int [B, T] check ...
        tt = np.zeros_like(ids) if token_type_ids is None else np.asarray(token_type_ids)  # default: segment A
        am = np.ones(ids.shape, dtype=bool) if attn_mask is None else np.asarray(attn_mask).astype(bool)
        # ... shape checks ...
        x = self.pos_emb(self.word_emb(ids)) + self.type_emb(tt)   # word + position + type
        x = self.drop(self.emb_norm(x))
        # Every query may read every real key, left and right: no causal mask.
        mask = am[:, None, None, :]
        for layer in self.layers:
            x = layer(x, mask)
        return x
```

`F.gelu` without `approximate` is the exact $u\,\Phi(u)$. It differs from GPT-2's tanh form by about $10^{-3}$, which matters for parity with Hugging Face.

### The MLM head

```python
class BertForMLM(Module):
    def __init__(self, cfg, rng=None):
        # ...
        self.bert = BertEncoder(cfg, r)
        self.transform = Linear(cfg.d_model, cfg.d_model, rng=r)
        # ... N(0, 0.02^2) init ...
        self.transform_norm = LayerNorm(cfg.d_model, cfg.ln_eps)
        # The decoder's weight IS word_emb.weight (tied); only its bias is new.
        self.decoder_bias = Tensor(np.zeros(cfg.vocab, dtype=np.float32), requires_grad=True)

    def forward(self, ids, token_type_ids=None, attn_mask=None, labels=None):
        h = self.bert(ids, token_type_ids, attn_mask)
        h = self.transform_norm(F.gelu(self.transform(h)))
        logits = F.matmul(h, F.transpose(self.bert.word_emb.weight, 0, 1)) + self.decoder_bias
        if labels is None:
            return logits, None
        # Only the picked positions carry a label; the rest are -100.
        return logits, cross_entropy(logits, np.asarray(labels).astype(np.int64), ignore_index=-100)
```

Keeping the encoder under the attribute `bert` gives `bert.`-prefixed keys, so a bare `BertEncoder` (used by ELECTRA and the fine-tuning heads) and the MLM model share one key mapping.

### The loader

`hf_bert_encoder_sd` is a table of renames, one per sublayer:

```python
    for i in range(n_layers):
        h, m = f"encoder.layer.{i}.", f"layers.{i}."
        # HF's BERT uses nn.Linear ([out, in]): no transpose, unlike GPT-2's Conv1D.
        for ours, theirs in (
            ("attn.q_proj", "attention.self.query"),
            ("attn.k_proj", "attention.self.key"),
            ("attn.v_proj", "attention.self.value"),
            ("attn.out_proj", "attention.output.dense"),
            ("attn_norm", "attention.output.LayerNorm"),
            ("ff1", "intermediate.dense"),
            ("ff2", "output.dense"),
            ("ff_norm", "output.LayerNorm"),
        ):
            out[f"{m}{ours}.weight"] = _hf(sd, f"{h}{theirs}.weight")
            out[f"{m}{ours}.bias"] = _hf(sd, f"{h}{theirs}.bias")
```

`load_hf_bert` dispatches on the module type. A `BertForMLM` gets the encoder keys prefixed with `bert.` plus `cls.predictions.transform.*` and `cls.predictions.bias`; a `BertEncoder` gets the encoder keys alone, and the head keys in the checkpoint are simply not read; anything else is a `TypeError`. `_hf` accepts keys with or without Hugging Face's `bert.` prefix. The full solution is in [`bert.py`](../../../../course/solutions/python/tinyllm/obj/bert.py).

### Why this design

Masking costs $O(BT)$ with at most three draws per position. The encoder is $O(L(T^2 d + T d\, d_{ff}))$, and the head adds $O(TVd)$, though only 15% of positions carry loss.

Masking is a **Python loop** rather than vectorized `rng` calls because vectorizing would draw an action uniform for every position, and the stream would no longer match Rust and Go. Determinism across languages is worth more than speed here, and masking is cheap next to the forward pass. Masking happens **at batch time** with a fresh `rng` per batch, not as a precomputed masked corpus, so each pass over the data sees new masks (RoBERTa's dynamic masking) for free. The **tied decoder** matches Hugging Face's checkpoints and halves the output parameters compared with a separate $V \times d$ matrix. The tradeoff accepted is that only 15% of positions produce a gradient signal; [ELECTRA](03-electra-replaced-token-detection.md) fixes that inefficiency.

## Testing it

A correct solution has to pass the following checks.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked mask | unit | the worked batch, draw for draw, with a generator that fails on any extra draw | everyone agrees on the op order |
| a kept token is still predicted | unit | a kept token is still labelled | the 10% that teaches "visible may be right" |
| the op order on the real generator | golden | a `[4, 32]` batch equals the op order replayed on the frozen PCG32 | the same batch in every language |
| masking rates | statistical | picked share within 4 sd of $p$; the exact shown shares by chi-square | the objective's proportions |
| agreement with Hugging Face BERT | golden | a random tiny `BertForMaskedLM`: logits, MLM loss, and gradients through the loader | real BERT checkpoints load (ELECTRA, fine-tuning heads) |
| one layer against the formula | differential | one layer in float64 numpy with the exact GELU and post-LN | the activation and the norm placement |
| padding is never read | property | ids at padded positions change no real output | batches of different lengths |
| reads both directions | property | the last token changes the first position | bidirectional, unlike GPT |
| token types are added | property | segment B changes the output; default is A | sentence-pair tasks |
| loss only at labelled positions | unit | the loss is the cross-entropy at the labelled position | no loss from visible tokens |
| the decoder is tied and has a bias | unit | parameter names and order; no decoder weight | checkpoint keys for ELECTRA and the zoo |
| batches from text | unit | WordPiece ids, padding, attention mask; specials never picked at $p = 1$ | batches straight from text |
| a bare encoder from an MLM checkpoint; roundtrip | unit | `load_hf_bert` into a `BertEncoder`; the model directory roundtrips | classifiers start from pretrained encoders |
| bad arguments | boundary | bad ids, shapes, rates, mask id, module type | errors, not silent garbage |

The model tests in [`test_bert.py`](../../../../course/solutions/learner-tests/L6.2/python/tests/l6-2-bert/test_bert.py) use two oracles: the op order replayed on a second `PCG32` from the same seed, and one layer plus the head in float64 numpy with `math.erf`.

## What can go wrong

Many masking bugs leave the rates looking right while changing what the model learns or which batch it sees. The common ones are these.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| special or padded positions picked | the model learns to predict [CLS] and [PAD] | `mlm_mask` skips `special_mask` positions before drawing; `mlm_batch` marks specials and padding |
| every position labelled | the loss is dominated by copying visible tokens | labels start at $-100$ and only picked positions are overwritten |
| random and keep swapped | the rates look right; the batch differs from every other implementation | `SPLIT = [0.8, 0.1, 0.1]` with action order `MASK, RANDOM, KEEP` |
| the label is what was shown, not the original | the model learns to predict [MASK] | `flat_lab[i] = flat_in[i]` before the input changes |
| a second uniform drawn for every position | a different stream from the second position on | the action uniform is drawn only after a position is picked |
| GPT-2's tanh GELU instead of the exact one | HF parity off by about $10^{-3}$ | `F.gelu` with the default exact form |
| pre-LN instead of BERT's post-LN | a different model; HF disagrees | `BertLayer` is `norm(x + sublayer(x))` |
| padding keys not masked | outputs depend on the batch's longest sentence | `mask = am[:, None, None, :]` in every layer |
| a causal mask copied from GPT | the encoder is one-directional | the mask is padding only |
| token types never added | sentence pairs look like one sentence | `+ self.type_emb(tt)` before the embedding LayerNorm |
| the decoder bias left out | HF disagrees | `decoder_bias` is added to the tied logits and loaded from `cls.predictions.bias` |

> **Warning:** Drawing the action uniform for every position, picked or not, is the easiest way to vectorize masking and the easiest way to break cross-language determinism. The rates still pass a statistical test; only a draw-for-draw golden comparison catches it.

## Where this leads

[ELECTRA](03-electra-replaced-token-detection.md) uses a small `BertForMLM` as its generator and a `BertEncoder` as its discriminator, both masked with `mlm_mask`. The [fine-tuning heads](05-fine-tuning-heads-and-the-linear-head-export.md) put sequence and token classification heads on `BertEncoder`, and the [model zoo](07-lm-evaluation-harness-and-the-model-zoo.md) reports its classification rows.

In production, Hugging Face's `DataCollatorForLanguageModeling` (in `transformers/data/data_collator.py`) masks at batch time and adds whole-word masking. RoBERTa replaced static masking with a new mask every epoch, and later work found higher rates, such as 40%, help large models. ModernBERT adds RoPE, GeGLU, alternating local and global attention, and an 8k context. And ELECTRA's replaced-token detection, the next section, puts a loss on every position instead of 15%.

Further reading:

- Devlin et al., "BERT" (NAACL 2019), section 3.1 and appendix A.1.
- Liu et al., "RoBERTa" (2019), section 4.1 (dynamic masking).
- Wettig et al. (2023), on 40% masking.
- Warner et al. (2024), ModernBERT.
- Clark et al. (2020), ELECTRA.

## Key terms

- **Masked language modeling (MLM)**: training an encoder to recover the original tokens at corrupted positions.
- **80/10/10 rule**: a picked position shows [MASK], a random token, or itself with probabilities 0.8, 0.1, 0.1.
- **Op order**: the fixed sequence of random draws that makes masking reproducible across implementations.
- **Post-LN**: normalization applied after the residual addition, $\text{LN}(x + f(x))$.
- **Token type embedding**: a learned vector added per segment (A or B) for sentence-pair inputs.

## Summary

- BERT keeps the 2017 encoder (post-LN, exact GELU) and reads in both directions; only padding is masked.
- The objective is fill-in-the-blank: pick 15% of the non-special positions, show [MASK] at 80%, a random token at 10%, the token itself at 10%, and predict the **original** token at exactly the picked positions.
- One fixed op order over one PCG32 stream makes masking reproducible in every implementation, and its rates are checked statistically against the exact shares.
- The MLM head transforms, normalizes, and reuses the word table as its decoder, with its own bias.
- Hugging Face's BERT uses `nn.Linear`, so loading is a pure rename.

## Review questions

1. Why must the label be written before the input is overwritten?
   <details><summary>Answer</summary>The label is the original token. Writing it after the [MASK] or random replacement would label the position with what was shown, and the model would learn to predict [MASK].</details>
2. Why does a picked position show [MASK] with probability $0.8 + 0.1/V$, not exactly 0.8?
   <details><summary>Answer</summary>The random action draws uniformly over all $V$ ids, which includes the [MASK] id with probability $1/V$. The same happens for the original token, so "itself" is shown with probability $0.1 + 0.1/V$.</details>
3. What changes in the random stream if the action uniform is drawn for every position, picked or not?
   <details><summary>Answer</summary>Every unpicked position consumes an extra draw, so from the second non-special position on, all later picks and actions come from different numbers. The rates still look right, but the batch differs from every other implementation of the op order.</details>
4. Why does `mlm_batch` truncate a long row as `ids[: max_len - 1] + ids[-1:]` instead of `ids[:max_len]`?
   <details><summary>Answer</summary>`encode(add_special=True)` ends every row with [SEP]. Plain slicing would cut it off, so long rows would end mid-sentence with no separator, unlike every row the model sees otherwise. Keeping the last id preserves the `[CLS] ... [SEP]` frame.</details>
5. Why does loading a BERT checkpoint need no transposes when GPT-2 needed them?
   <details><summary>Answer</summary>Hugging Face's BERT is written with `nn.Linear`, which stores weights as `[out, in]` like the solution's `Linear`. GPT-2 uses `Conv1D`, which stores `[in, out]`.</details>
