# BERT encoder and MLM masking

## Overview

| | |
|---|---|
| **Module** | `L6.2` · build · Python · Pass 5 · 1.5 h reading |
| **Solution** | [`python/tinyllm/obj/bert.py`](../../../../course/solutions/python/tinyllm/obj/bert.py): `mlm_mask`, `special_ids`, `mlm_batch`, `BertConfig`, `BertLayer`, `BertEncoder`, `BertForMLM`, `hf_bert_encoder_sd`, `load_hf_bert`, `save_bert`, `load_bert`; model oracle tests in [`learner-tests/L6.2/.../test_bert.py`](../../../../course/solutions/learner-tests/L6.2/python/tests/l6-2-bert/test_bert.py) |
| **Contract** | [`course/contracts/py/tinyllm/obj/bert.pyi`](../../../../course/contracts/py/tinyllm/obj/bert.pyi) |
| **Needs** | `L5.3` attention · `L5.4` `LearnedPE` · `L1.3` `WordPieceTokenizer` · `M07.1` `sample_categorical` · `M06.3` `PCG32` · `L0.1` · `L0.2` · `L0.3` `cross_entropy` · `L0.4` layers · `L0.6` safetensors · `M07.3` `normal_init` |
| **Used by** | `L6.3` ELECTRA's generator and discriminator · `L6.5`'s classification heads · `L6.7` the zoo's classification rows |
| **Optional depth** | Devlin et al., "BERT" (NAACL 2019), sections 3.1 and appendix A.1; Liu et al., "RoBERTa" (2019), section 4.1 (dynamic masking) |

## Key Takeaways

- BERT keeps `L5.5`'s encoder (post-LN, exact GELU) and reads in both directions: only padding is masked.
- The objective is fill-in-the-blank: pick 15% of the non-special positions, show [MASK] at 80% of them, a random token at 10%, the token itself at 10%, and predict the **original** token at exactly the picked positions.
- One fixed op order over one PCG32 stream makes masking reproducible in every implementation, and its rates are checked statistically.
- The MLM head transforms, normalizes, and reuses the word table as its decoder, with its own bias.

## How to read this chapter

Sections 1 to 3 derive the masked-LM objective, fix the masking op order, and work one batch by hand draw for draw. Section 4 fixes the interface. Section 5 walks through the solution and explains why each choice is right. Try to predict each step before reading it.

---

## 1. Why now

The GPT of `L6.1` reads left to right, which is what generation needs and what classification does not: to decide whether "not bad at all" is positive, every word should see every other. A bidirectional encoder cannot be trained by next-token prediction, because each position could simply read the answer. BERT's fix is to hide some tokens and ask for them back. The system needs this encoder twice: ELECTRA (`L6.3`) trains a small BERT as its generator and a BERT-shaped discriminator, and the classification heads of `L6.5` (the usage-policy head the gateway evaluates) sit on top of it. Its tokenizer is the WordPiece of `L1.3`.

## 2. Principles

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

### 2.1 The encoder

$$h_0 = \text{LN}(W_{word}[x] + W_{pos}[0..T-1] + W_{type}[\text{tt}]),$$

then $L$ post-LN layers: $h \leftarrow \text{LN}(h + \text{Attn}(h))$, $h \leftarrow \text{LN}(h + W_2\,\text{GELU}(W_1 h))$, with the exact GELU $u\,\Phi(u)$ (Hugging Face's `"gelu"`) and LayerNorm $\epsilon = 10^{-12}$. The attention mask is padding only, `[B, 1, 1, T]`: every real position reads every other real position, left and right. Changing the last word changes the first position's output, which a GPT can never do.

### 2.2 Masking 15%, and 80/10/10

Visit positions in C order (row by row):

1. a special position draws nothing; label $-100$;
2. draw $u$; if $u \ge p$ the position is not picked; label $-100$;
3. otherwise label $y_t = x_t$ (the original), draw $a$ = `sample_categorical([0.8, 0.1, 0.1], u')` (`M07.1`, inverse CDF);
4. $a = 0$: show [MASK]; $a = 1$: show `rng.below(V)`, a uniform random id; $a = 2$: show $x_t$ itself.

Why not always [MASK]? [MASK] never appears when the encoder is fine-tuned or used, so a model trained only on [MASK] positions would learn that real tokens need no thought. The random 10% teaches it that any visible token may be wrong, and the kept 10% that a visible token may be right. The label is always the original token, also at kept positions: those positions are predictions too. Draws happen only where the steps say, so the same seed gives the same batch in Python, Rust, and Go.

Because the random id is uniform over all $V$ ids, it can be [MASK] or the original token by chance: the exact shares of what a picked position **shows** are $0.8 + 0.1/V$ ([MASK]), $0.1 (V - 2)/V$ (another token), and $0.1 + 0.1/V$ (itself), which is what a statistical check of the rates must use.

### 2.3 The MLM head and loss

$$z_t = \text{LN}(\text{GELU}(W_T h_t + b_T))\,W_{word}^\top + b_{dec}, \qquad \mathcal{L} = \frac{1}{|\{t : y_t \ne -100\}|} \sum_{y_t \ne -100} -\log \operatorname{softmax}(z_t)_{y_t}.$$

The decoder weight is the word table (tied); only its bias $b_{dec}$ is new. Positions that were not picked contribute no loss and no gradient (`L0.3`'s `ignore_index`).

### 2.4 Hugging Face names

HF's BERT uses `nn.Linear`, so no transpose (unlike GPT-2's Conv1D, `L6.1`): `embeddings.{word,position,token_type}_embeddings`, `embeddings.LayerNorm`, `encoder.layer.i.attention.self.{query,key,value}`, `attention.output.dense`, `attention.output.LayerNorm`, `intermediate.dense`, `output.dense`, `output.LayerNorm`, and the head `cls.predictions.transform.dense`, `transform.LayerNorm`, `cls.predictions.bias`. HF's attention mask adds the most negative float instead of $-\infty$; a padded key gets weight 0 either way.

## 3. Worked example by hand

Ids `[CLS] the cat sat [SEP] [PAD]` = $(2, 5, 6, 7, 3, 0)$, [MASK] = 4, $V = 40$, $p = 0.15$, and a generator that will return the uniforms $0.05, 0.50, 0.90, 0.10, 0.85$ and then `below(40)` = 9.

| position | token | special? | draws | picked? | action | shown | label |
|---|---|---|---|---|---|---|---|
| 0 | [CLS] | yes | none | | | 2 | -100 |
| 1 | the | no | $u = 0.05 < 0.15$; $a$: $0.50 < 0.8$ | yes | [MASK] | 4 | 5 |
| 2 | cat | no | $u = 0.90$ | no | | 6 | -100 |
| 3 | sat | no | $u = 0.10$; $a$: $0.8 \le 0.85 < 0.9$; below = 9 | yes | random | 9 | 7 |
| 4 | [SEP] | yes | none | | | 3 | -100 |
| 5 | [PAD] | yes | none | | | 0 | -100 |

Inputs $(2, 4, 6, 9, 3, 0)$, labels $(-100, 5, -100, 7, -100, -100)$; exactly five uniforms and one `below`.

## 4. The interface

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked mask | unit | section 3, draw for draw, with a generator that fails on any extra draw | everyone agrees on the op order |
| a kept token is still predicted | unit | a kept token is still labelled | the 10% that teaches "visible may be right" |
| the op order on the real generator | golden | a `[4, 32]` batch equals the op order replayed on the frozen PCG32 | the same batch in every language |
| masking rates | statistical | picked share within 4 sd of $p$; the shown shares of section 2.2 by chi-square | the objective's proportions |
| agreement with Hugging Face BERT | golden | a random tiny `BertForMaskedLM`: logits, MLM loss, and gradients through the loader | real BERT checkpoints load (`L6.3`, `L6.5`) |
| one layer against the formula | differential | one layer in float64 numpy with the exact GELU and post-LN | the activation and the norm placement |
| padding is never read | property | ids at padded positions change no real output | batches of different lengths |
| reads both directions | property | the last token changes the first position | bidirectional, unlike GPT |
| token types are added | property | segment B changes the output; default is A | sentence-pair tasks |
| loss only at labelled positions | unit | the loss is the cross-entropy at the labelled position | no loss from visible tokens |
| the decoder is tied and has a bias | unit | parameter names and order; no decoder weight | checkpoint keys for `L6.3` and the zoo |
| batches from text | unit | WordPiece ids, padding, attention mask; specials never picked at $p = 1$ | batches straight from text |
| a bare encoder from an MLM checkpoint; roundtrip | unit | `load_hf_bert` into a `BertEncoder`; the model directory roundtrips | classifiers start from pretrained encoders |
| bad arguments | boundary | bad ids, shapes, rates, mask id, module type | errors, not silent garbage |

A model test suite with both oracles (the op order replayed on a second `PCG32` from the same seed, and one layer plus the head in float64 numpy with `math.erf`) is in [`test_bert.py`](../../../../course/solutions/learner-tests/L6.2/python/tests/l6-2-bert/test_bert.py).

## 5. The solution, step by step

### 5.1 From first principles

1. Classification wants every token to see every other, so the encoder attends both ways and masks only padding. Then next-token prediction is impossible (each position can read the answer), so the objective must hide information: corrupt some inputs and predict the originals.
2. Only corrupted positions are informative, so only they are labelled; all others get $-100$ and drop out of the loss. Special positions ([CLS], [SEP], [PAD]) are never picked: predicting them teaches nothing about language.
3. If the corruption were always [MASK], the encoder could ignore every visible token, and [MASK] never appears downstream. Mixing in random tokens (10%) and the token itself (10%) forces the model to judge every visible token. The label is always the **original**, including at kept positions.
4. Reproducibility across languages demands a fixed draw order: positions in C order, specials draw nothing, one uniform to pick, and only for picked positions one uniform for the action and, for a random action, one `below(V)`. In the worked example that is exactly five uniforms and one `below`.
5. The architecture is `L5.5`'s encoder in post-LN with BERT's details: exact GELU, three summed embeddings (word, position, type) followed by a LayerNorm, $\epsilon = 10^{-12}$.
6. The MLM head reuses the word table as its output matrix, so only a dense transform, a LayerNorm and a bias are new. HF uses `nn.Linear`, so the loader is a pure rename with no transposes.

### 5.2 The code, annotated

**Masking.** One loop over the flattened batch; every draw sits exactly where the op order says:

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
        action = sample_categorical(SPLIT, rng.uniform())  # draw 2 (M07.1, inverse CDF)
        if action == MASK:
            flat_in[i] = mask_id
        elif action == RANDOM:
            flat_in[i] = rng.below(vocab)      # draw 3, only for RANDOM
        # KEEP: the input shows the token itself, but it is still predicted.
    return inputs, labels
```

The label is written before the input is overwritten, which is what makes it the original rather than what was shown.

**Batches from text.** `mlm_batch` tokenizes, truncates keeping `[SEP]` last, pads, and marks as special every special id and every padded position:

```python
    special = np.isin(ids, special_ids(tok)) | ~attn   # specials and padding never picked
    vocab = max(tok.vocab.values()) + 1
    inputs, labels = mlm_mask(ids, special, mask_id, vocab, p, rng)
```

**The layer and the encoder.**

```python
class BertLayer(Module):
    # ... attn, attn_norm, ff1, ff2, ff_norm; every Linear at N(0, 0.02^2) ...
    def forward(self, x: Tensor, mask: NDArray) -> Tensor:
        # Post-LN, as the 2017 transformer (L5.5's norm="post"), exact GELU.
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

`F.gelu` without `approximate` is the exact $u\,\Phi(u)$; that differs from GPT-2's tanh form by about $10^{-3}$, which matters for HF parity.

**The MLM head.**

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

Keeping the encoder under the attribute `bert` gives `bert.`-prefixed keys, so a bare `BertEncoder` (for `L6.3` and `L6.5`) and the MLM model share one key mapping.

**The loader.** `hf_bert_encoder_sd` is a table of renames, one per sublayer:

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

`load_hf_bert` dispatches on the module type: a `BertForMLM` gets the encoder keys prefixed with `bert.` plus `cls.predictions.transform.*` and `cls.predictions.bias`; a `BertEncoder` gets the encoder keys alone (the head keys in the checkpoint are simply not read); anything else is a `TypeError`. `_hf` accepts keys with or without HF's `bert.` prefix.

Full solution: [`bert.py`](../../../../course/solutions/python/tinyllm/obj/bert.py).

### 5.3 Why this solution

- **Cost.** Masking is $O(BT)$ with at most three draws per position. The encoder is $O(L(T^2 d + T d\, d_{ff}))$; the head adds $O(TVd)$, though only 15% of positions carry loss.
- **A Python loop for masking** rather than vectorized `rng` calls: vectorizing would draw an action uniform for every position (pitfall 5), so the stream would no longer match Rust and Go. Determinism across languages is worth more than speed here; masking is cheap next to the forward pass.
- **Static masking at batch time** with a fresh `rng` per batch, not a precomputed masked corpus: each pass over the data sees new masks (RoBERTa's dynamic masking) for free.
- **Tied decoder** rather than a separate $V \times d$ matrix: matches HF's checkpoints and halves the output parameters.
- **Tradeoff accepted.** Only 15% of positions produce a gradient signal; ELECTRA (`L6.3`) fixes that inefficiency.

### 5.4 Explain it back

1. Why must the label be written before the input is overwritten?
   <details><summary>Answer</summary>The label is the original token. Writing it after the [MASK] or random replacement would label the position with what was shown, and the model would learn to predict [MASK].</details>
2. Why does a picked position show [MASK] with probability $0.8 + 0.1/V$, not exactly 0.8?
   <details><summary>Answer</summary>The random action draws uniformly over all $V$ ids, which includes the [MASK] id with probability $1/V$. The same happens for the original token, so "itself" is shown with probability $0.1 + 0.1/V$.</details>
3. What changes in the random stream if the action uniform is drawn for every position, picked or not?
   <details><summary>Answer</summary>Every unpicked position consumes an extra draw, so from the second non-special position on, all later picks and actions come from different numbers. The rates still look right, but the batch differs from every other implementation of the op order.</details>
4. Why does `mlm_batch` truncate a long row as `ids[: max_len - 1] + ids[-1:]` instead of `ids[:max_len]`?
   <details><summary>Answer</summary>`encode(add_special=True)` ends every row with [SEP]. Plain slicing would cut it off, so long rows would end mid-sentence with no separator, unlike every row the model sees otherwise. Keeping the last id preserves the `[CLS] ... [SEP]` frame.</details>
5. Why does loading a BERT checkpoint need no transposes when GPT-2 needed them?
   <details><summary>Answer</summary>HF's BERT is written with `nn.Linear`, which stores weights as `[out, in]` like the solution's `Linear`. GPT-2 uses `Conv1D`, which stores `[in, out]`.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. special or padded positions picked | the model learns to predict [CLS] and [PAD] | `mlm_mask` skips `special_mask` positions before drawing; `mlm_batch` marks specials and padding |
| 2. every position labelled | the loss is dominated by copying visible tokens | labels start at $-100$ and only picked positions are overwritten |
| 3. random and keep swapped | the rates look right; the batch differs from every other implementation | `SPLIT = [0.8, 0.1, 0.1]` with action order `MASK, RANDOM, KEEP` |
| 4. the label is what was shown, not the original | the model learns to predict [MASK] | `flat_lab[i] = flat_in[i]` before the input changes |
| 5. a second uniform drawn for every position | a different stream from the second position on | the action uniform is drawn only after a position is picked |
| 6. GPT-2's tanh GELU instead of the exact one | HF parity off by about $10^{-3}$ | `F.gelu` with the default exact form |
| 7. pre-LN instead of BERT's post-LN | a different model; HF disagrees | `BertLayer` is `norm(x + sublayer(x))` |
| 8. padding keys not masked | outputs depend on the batch's longest sentence | `mask = am[:, None, None, :]` in every layer |
| 9. a causal mask copied from GPT | the encoder is one-directional | the mask is padding only |
| token types never added | sentence pairs look like one sentence | `+ self.type_emb(tt)` before the embedding LayerNorm |
| the decoder bias left out | HF disagrees | `decoder_bias` is added to the tied logits and loaded from `cls.predictions.bias` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L5.3` | the attention of every layer |
| Back | `L5.4` | `LearnedPE` is the position embedding |
| Back | `L1.3` | `WordPieceTokenizer` encodes text in `mlm_batch` |
| Back | `M07.1` | `sample_categorical` picks the 80/10/10 action |
| Back | `M06.3` | `PCG32`: every draw of the masking, and the init stream |
| Back | `L0.1` | `Tensor` |
| Back | `L0.2` | exact `gelu`, `matmul` for the tied decoder |
| Back | `L0.3` | `cross_entropy` with `ignore_index` |
| Back | `L0.4` | `Linear`, `Embedding`, `LayerNorm` |
| Back | `L0.6` | safetensors model directory |
| Back | `M07.3` | `normal_init` |
| Forward | `L6.3` | ELECTRA: a small `BertForMLM` generator and a `BertEncoder` discriminator, masked with `mlm_mask` |
| Forward | `L6.5` | sequence and token classification heads on `BertEncoder` |
| Forward | `L6.7` | the zoo's classification rows |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `mlm_mask` | HF `DataCollatorForLanguageModeling` | masking at batch time, whole-word masking | `transformers/data/data_collator.py` |
| static 15% | RoBERTa dynamic masking; 40% masking | a new mask every epoch; higher rates for large models | Liu et al. 2019; Wettig et al. 2023 |
| `BertEncoder` | ModernBERT | RoPE, GeGLU, alternating local and global attention, 8k context | Warner et al. 2024 |
| MLM | ELECTRA's replaced-token detection | a loss on every position, not 15% | `L6.3`; Clark et al. 2020 |
