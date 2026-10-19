# ELECTRA replaced-token detection

## Overview

| | |
|---|---|
| **Module** | `L6.3` · build · Python · Pass 5 · 1.5 h reading |
| **Solution** | [`python/tinyllm/obj/electra.py`](../../../../course/solutions/python/tinyllm/obj/electra.py): `ElectraDiscriminatorHead`, `ElectraDiscriminator`, `ELECTRA`, `replace_tokens`, `rtd_loss`, `electra_step`, `rtd_accuracy`, `save_electra`, `load_electra`; model oracle tests in [`learner-tests/L6.3/.../test_electra.py`](../../../../course/solutions/learner-tests/L6.3/python/tests/l6-3-electra/test_electra.py) |
| **Contract** | [`course/contracts/py/tinyllm/obj/electra.pyi`](../../../../course/contracts/py/tinyllm/obj/electra.pyi) |
| **Needs** | `L6.2` `BertEncoder`, `BertForMLM`, `mlm_mask` · [`M07.1` `sample_categorical`](../../mathematics-for-computing/07-probability-statistics/01-categorical-sampling.md) · `L0.3` `bce_with_logits` · `L0.1` · `L0.2` · `L0.4` · `L0.6` safetensors · `M06.3` PCG32 · `M07.3` `normal_init` |
| **Used by** | `L6.7` the zoo's ELECTRA row (replaced-token detection accuracy) · the discriminator body is the alternative classification backbone of `L6.5` (`tl_arch = "electra"`): at the end of Part 6 the system trains ELECTRA, then a LoRA classifier over its discriminator |
| **Optional depth** | Clark, Luong, Le, and Manning, "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators" (ICLR 2020), sections 2, 3.2, and appendix A; Goodfellow et al., "Generative Adversarial Nets" (2014), for what ELECTRA is not |

## Key Takeaways

- The discriminator learns from **every** real token, not only the 15% BERT masks: each one is labelled original or replaced.
- A sampled token equal to the original is labelled **original**.
- The sample is data: the generator trains on its MLM loss only, and the total is $L_G + \lambda L_D$ with $\lambda = 50$.
- The discriminator is BERT's encoder plus a dense, GELU, dense head, and its loss averages over real tokens only; both match Hugging Face.
- Generator and discriminator share one set of embeddings.

## How to read this chapter

Sections 1 to 3 derive replaced-token detection from BERT's masking, fix the labels and the draw order, and work one step by hand. Section 4 fixes the interface. Section 5 walks through the solution and explains why each choice is right. Try to predict each step before reading it.

---

## 1. Why now

The BERT of `L6.2` learns from a masked-LM loss on 15% of the tokens of each batch; the other 85% cost a full forward and backward pass and teach nothing. On the small corpus and compute budget of this course that waste shows: a BERT pretrained for the Part 6 step budget is a weak backbone for the sentiment classifier of `L6.5`. ELECTRA turns every position into a training signal with the same encoder: a small generator fills the masked positions with plausible guesses, and the encoder, now a discriminator, says for every token whether it was replaced. The zoo (`L6.7`) puts the two backbones side by side on the same classification task, which is the comparison the paper is about.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x$ | the batch of token ids | `int64[B, T]` |
| $m$ | the real-token mask (`attn_mask`), 1 = real, 0 = padding | `bool[B, T]` |
| $\tilde x$, $\ell$ | the masked inputs and MLM labels from `mlm_mask` (`-100` = not picked) | `int64[B, T]` |
| $G$ | the generator, a small `BertForMLM` | Module |
| $D$ | the discriminator, `BertEncoder` plus a token head | Module |
| $p_G(\cdot \mid \tilde x, t)$ | the generator's distribution at position $t$ | `float64[V]` |
| $\hat x$ | the corrupted ids (`corrupt`) | `int64[B, T]` |
| $y_t = [\hat x_t \ne x_t]$ | the discriminator's label (`is_replaced`) | `float32[B, T]` |
| $L_G$, $L_D$ | MLM cross-entropy, replaced-token BCE | scalars |
| $\lambda$ | the discriminator's loss weight, 50 | `float` |

### 2.1 One step

1. Mask like BERT: $(\tilde x, \ell) = $ `mlm_mask(x, special` $\lor \lnot m$ `, mask_id, V, p, rng)`. Padding counts as special, so it is never picked.
2. Run the generator on $\tilde x$: logits and $L_G$, the mean cross-entropy over the picked positions.
3. At each picked position, in C order, draw one uniform and sample $\hat x_t \sim p_G(\cdot \mid \tilde x, t)$ with `sample_categorical` (`M07.1`, inverse CDF in float64); elsewhere $\hat x_t = x_t$.
4. Label every token: $y_t = 1$ when $\hat x_t \ne x_t$. Run the discriminator on $\hat x$ and take $L_D$, the mean binary cross-entropy over the real tokens.
5. $L = L_G + \lambda L_D$.

The random draws are `mlm_mask`'s, then one uniform per picked position, all from one generator. Sampling is not differentiable, and the paper does not try to make it so (an adversarial generator trained by reinforcement learning did worse): the corrupted ids enter the discriminator as plain data, so $L_D$ sends no gradient into $G$. Each network learns from its own loss, and one backward pass of $L$ does both.

### 2.2 The discriminator's labels

The label asks "is this token different from the original?", not "was this position masked?". A good generator often samples the original token back, especially for function words; those positions are labelled original. Labelling every masked position as replaced teaches the discriminator to flag tokens that never changed, and its accuracy then measures the masking rate, not the text.

### 2.3 The discriminator

The body is `L6.2`'s `BertEncoder`, unchanged, under the name `electra`. The head is Hugging Face's `ElectraDiscriminatorPredictions`: `dense` ($d \to d$), the exact GELU, `dense_prediction` ($d \to 1$), and the last axis dropped, so the output is one logit per token, positive meaning "replaced". The loss is `L0.3`'s `bce_with_logits` over the real tokens only, the mean over them: padding contributes neither to the sum nor to the count. With these names, a Hugging Face ELECTRA checkpoint loads through `L6.2`'s `hf_bert_encoder_sd` after dropping its `electra.` prefix.

### 2.4 Sharing embeddings

ELECTRA ties the generator's token, position, and type embeddings to the discriminator's: one Tensor under two names. The generator is otherwise smaller (fewer layers here; the paper also narrows it), which matters: a generator as strong as the discriminator produces replacements too hard to detect. In the `ELECTRA` module the discriminator registers first, so the shared Tensors are listed once, under `discriminator.`, and both losses add their gradients into them.

The model directory holds both networks, the tie, the mask id, and the special ids, so the zoo can corrupt held-out text the same way and report replaced-token detection accuracy (`rtd_accuracy`): over the real tokens, how often the discriminator's call (logit $> 0$) matches $y$.

## 3. Worked example by hand

Four tokens $x = (5, 6, 7, 8)$; `mlm_mask` picked positions 1 and 3, so $\ell = (-100, 6, -100, 8)$. The generator's distributions at the picked positions:

| position | $p_G$ | uniform | cumulative sum crosses it at | $\hat x_t$ | $y_t$ |
|---|---|---|---|---|---|
| 1 | 0.75 on token 6, 0.25 on token 9 | 0.1 | token 6 (0.75) | 6 | 0 (the original) |
| 3 | 0.5 on token 2, 0.5 on token 8 | 0.3 | token 2 (0.5) | 2 | 1 |

So $\hat x = (5, 6, 7, 2)$ and $y = (0, 0, 0, 1)$. Position 1 was masked, but the sample is the original token: not replaced.

With all discriminator logits 0, each token's BCE is $\ln 2$ and $L_D = \ln 2 = 0.693147$. With logits $(-2, -2, -2, 2)$ every call is right and each token pays $\ln(1 + e^{-2}) = 0.126928$, the mean too.

## 4. The interface

```python
class ElectraDiscriminator(Module):            # electra.* (BertEncoder) + discriminator_predictions.*
    def __init__(self, cfg: BertConfig, rng=None) -> None: ...
    def forward(self, ids, token_type_ids=None, attn_mask=None) -> Tensor: ...    # [B, T] logits
class ELECTRA(Module):                         # discriminator.*, generator.* (tied embeddings)
    def __init__(self, gen_cfg: BertConfig, disc_cfg: BertConfig, rng=None, tie_embeddings=True) -> None: ...
def replace_tokens(ids, labels, gen_logits, rng, ignore_index=-100) -> tuple[NDArray, NDArray]: ...
def rtd_loss(disc_logits: Tensor, is_replaced, attn_mask=None) -> Tensor: ...
def electra_step(gen, disc, ids, rng, lam=50.0, *, mask_id, vocab=None, special_mask=None,
                 token_type_ids=None, attn_mask=None, p=0.15) -> dict: ...
def rtd_accuracy(model: ELECTRA, ids, rng, *, mask_id, ...) -> NDArray: ...
def save_electra(model, dir, mask_id, special_ids=(), tokenizer="file") -> None: ...
def load_electra(dir) -> tuple[ELECTRA, dict]: ...
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked labels | unit | section 3: $\hat x$, $y$, two uniforms used, both loss values | everyone agrees on the labels |
| a lucky sample is original | boundary | a generator certain of the original replaces nothing | the label is "different", not "masked" |
| one uniform per picked position | unit | three picked positions, three uniforms, C order | the same seed gives the same corruption everywhere |
| agreement with Hugging Face | golden | logits, loss, and gradients of HF's `ElectraForPreTraining` | pretrained ELECTRA checkpoints load and score alike |
| the loss ignores padding | property | padded logits and labels change nothing | padded batches train the same as unpadded ones |
| the step is its pieces | differential | the step equals mask, generate, sample, discriminate, recomputed from a copied rng | the order of the five steps |
| padding never masked or scored | property | padding and specials are never picked or replaced | no learning from padding |
| no gradient through the sample | property | generator gradients of $L$ equal those of $L_G$ alone | each network learns from its own loss |
| $\lambda$ weights the discriminator | unit | default $\lambda = 50$; $L = L_G + \lambda L_D$ for three values | the paper's balance of the two losses |
| tied embeddings are one tensor | property | one Tensor, listed once under `discriminator.`; mismatched configs raise | shared embeddings, saved once |
| save and load | property | the directory restores both networks, the tie, the mask id, and the specials | the zoo loads it |
| the accuracy metric | unit | an "always original" discriminator scores the fraction not replaced | the zoo's ELECTRA metric |

A model test suite (hand-made generator logits whose samples are known, BCE written out with `math.log1p`, the step recomputed from its pieces with a deep copy of the generator) is in [`test_electra.py`](../../../../course/solutions/learner-tests/L6.3/python/tests/l6-3-electra/test_electra.py).

## 5. The solution, step by step

### 5.1 From first principles

1. BERT spends a full forward and backward pass on every token but gets a signal from 15%. To get a signal from every token, every token needs a label. "Was this token changed?" is a question with an answer at every position.
2. For the question to be hard, the changes must be plausible. A small masked LM (`L6.2`'s `BertForMLM`) supplies plausible replacements: mask as BERT does, then sample from the generator at the picked positions.
3. The label must be defined by the result, not by the process: $y_t = [\hat x_t \ne x_t]$. Otherwise a lucky sample (position 1 in the worked example) teaches the discriminator to flag an unchanged token.
4. Sampling is discrete and not differentiable. Rather than invent a gradient, treat the sample as data: pass `gen_logits.data` (a plain array) to the sampler. Then $L_D$ cannot reach $G$, and $L = L_G + \lambda L_D$ trains each network on its own loss in one backward pass.
5. $L_D$ is a mean over real tokens of a binary cross-entropy, while $L_G$ is a mean over 15% of them with a $V$-way cross-entropy; the per-token scale of $L_D$ is much smaller, which is why the paper weights it by $\lambda = 50$.
6. The discriminator must read the **corrupted** ids $\hat x$, not the masked inputs $\tilde x$, or the task degenerates into "find the [MASK] tokens".
7. Matching HF's module names (`electra`, `discriminator_predictions.dense`, `dense_prediction`) lets a pretrained checkpoint load through `L6.2`'s BERT key mapping.
8. Sharing the embeddings means one Tensor object referenced by both networks; registering the discriminator first lists it under `discriminator.` in the `state_dict`.

### 5.2 The code, annotated

**The discriminator head.** Dense, exact GELU, dense to one logit, last axis dropped:

```python
class ElectraDiscriminatorHead(Module):
    def __init__(self, d_model: int, rng: Any = None) -> None:
        super().__init__()
        r = _init_rng(rng)
        self.dense = Linear(d_model, d_model, rng=r)
        self.dense.weight.data = normal_init((d_model, d_model), INIT_STD, r).astype(np.float32)
        self.dense_prediction = Linear(d_model, 1, rng=r)
        self.dense_prediction.weight.data = normal_init((1, d_model), INIT_STD, r).astype(np.float32)

    def forward(self, h: Tensor) -> Tensor:
        z = self.dense_prediction(F.gelu(self.dense(h)))  # [..., 1]
        return F.reshape(z, z.shape[:-1])                  # [B, T]: one logit per token
```

```python
class ElectraDiscriminator(Module):
    def __init__(self, cfg: BertConfig, rng: Any = None) -> None:
        # ...
        self.electra = BertEncoder(cfg, r)          # L6.2's encoder, unchanged
        self.discriminator_predictions = ElectraDiscriminatorHead(cfg.d_model, r)
```

**Tying.** The discriminator registers first; then the generator's three embedding attributes are pointed at the discriminator's Tensors:

```python
        self.discriminator = ElectraDiscriminator(disc_cfg, r)
        self.generator = BertForMLM(gen_cfg, r)
        if tie_embeddings:
            # One Tensor under two names: listed once, under the discriminator's.
            g, d = self.generator.bert, self.discriminator.electra
            g.word_emb.weight = d.word_emb.weight
            g.pos_emb.weight = d.pos_emb.weight
            g.type_emb.weight = d.type_emb.weight
```

Before this, the constructor raises if the vocabularies differ, or if tying is asked for with different `d_model`, `max_len` or `type_vocab`: tables of different shapes cannot be one Tensor. The generator's MLM decoder is tied to `g.word_emb.weight` (`L6.2`), so after tying it decodes with the shared table too.

**Sampling replacements.**

```python
def replace_tokens(ids, labels, gen_logits, rng, ignore_index=-100):
    x = np.asarray(ids)
    lab = np.asarray(labels)
    z = np.asarray(gen_logits, dtype=np.float64)   # float64 for the inverse CDF
    # ... shape checks ...
    corrupt = x.astype(np.int64).copy()
    flat_c, flat_l = corrupt.reshape(-1), lab.reshape(-1)
    rows = z.reshape(-1, z.shape[-1])
    for i in range(flat_c.size):  # C order, one uniform per masked position
        if flat_l[i] == ignore_index:
            continue
        w = np.exp(rows[i] - rows[i].max())        # softmax, shifted for stability
        flat_c[i] = sample_categorical(w / w.sum(), rng.uniform())  # M07.1
    # Replaced means DIFFERENT from the original: a lucky sample is original.
    return corrupt, (corrupt != x).astype(np.float32)
```

**The loss over real tokens.**

```python
def rtd_loss(disc_logits, is_replaced, attn_mask=None):
    y = np.asarray(is_replaced, dtype=np.float32)
    # ... shape checks ...
    real = np.ones(y.shape, dtype=bool) if attn_mask is None else np.asarray(attn_mask).astype(bool)
    # Mean over the real tokens only, as HF's ElectraForPreTraining.
    return bce_with_logits(disc_logits[real], y[real])
```

Indexing with the boolean mask before the mean makes padding drop out of both the sum and the count. Multiplying the per-token loss by the mask and dividing by $BT$ would instead make short sequences weigh less.

**One step.**

```python
def electra_step(gen, disc, ids, rng, lam=50.0, *, mask_id, vocab=None, special_mask=None,
                 token_type_ids=None, attn_mask=None, p=0.15):
    x = np.asarray(ids)
    real = np.ones(x.shape, dtype=bool) if attn_mask is None else np.asarray(attn_mask).astype(bool)
    special = np.zeros(x.shape, dtype=bool) if special_mask is None else np.asarray(special_mask, dtype=bool)
    V = vocab if vocab is not None else gen.bert.cfg.vocab
    # 1. mask like BERT (padding counts as special: never picked).
    inputs, labels = mlm_mask(x, special | ~real, mask_id, V, p, rng)
    # 2. the generator's MLM loss on the picked positions.
    gen_logits, gen_loss = gen(inputs, token_type_ids, attn_mask, labels)
    # 3. sample replacements: data only, no gradient through the choice.
    corrupt, is_replaced = replace_tokens(x, labels, gen_logits.data, rng)
    # 4. the discriminator labels every real token of the CORRUPTED ids.
    disc_logits = disc(corrupt, token_type_ids, attn_mask)
    disc_loss = rtd_loss(disc_logits, is_replaced, real)
    return {
        "loss": gen_loss + disc_loss * lam,     # 5. lambda on the discriminator's loss
        "gen_loss": gen_loss, "disc_loss": disc_loss, "disc_logits": disc_logits,
        "inputs": inputs, "labels": labels, "corrupt": corrupt, "is_replaced": is_replaced,
    }
```

`replace_tokens` receives the original `x` (not `inputs`), so unpicked positions keep their original ids and `is_replaced` compares against the true text. The same `rng` serves `mlm_mask` and the sampler, so the whole step is a single reproducible stream.

**The metric.** `rtd_accuracy` runs the step under `no_grad` and compares the discriminator's call with the truth on real tokens:

```python
    said = out["disc_logits"].data > 0  # logit > 0: "replaced"
    right = said == (out["is_replaced"] > 0.5)
    return right[real].astype(np.float64)    # per-token 0/1; the caller averages
```

`save_electra` writes the discriminator config under HF's key names, the generator config under `tl_generator`, plus `tl_tie_embeddings`, `tl_mask_id` and `tl_special_ids`, so the zoo can rebuild both networks and corrupt held-out text the same way.

Full solution: [`electra.py`](../../../../course/solutions/python/tinyllm/obj/electra.py).

### 5.3 Why this solution

- **Cost.** One step is a generator forward and backward (small), a discriminator forward and backward (full size), and $O(\text{picked} \cdot V)$ for sampling. Compared with BERT the discriminator pass is the same cost but every real token contributes gradient.
- **Sample as data** rather than a straight-through or REINFORCE gradient into the generator: the paper found the adversarial version worse, and data-only sampling keeps the two objectives independent and the gradient exact.
- **One shared `rng`** rather than separate streams for masking and sampling: one seed fixes the batch, and the draw order (mask draws, then one uniform per picked position) is a spec any language can replay.
- **Tied embeddings** rather than two tables: the generator's embeddings get the discriminator's gradient, and the checkpoint stores them once.
- **Tradeoff accepted.** The generator must be weaker than the discriminator to keep the task informative; this solution uses fewer layers, while the paper also narrows it (which would need HF's `embeddings_project`).

### 5.4 Explain it back

1. Why is the label $[\hat x_t \ne x_t]$ and not "position $t$ was masked"?
   <details><summary>Answer</summary>A generator often samples the original back. Labelling those positions "replaced" asks the discriminator to flag tokens that are identical to the truth, which it can only do by guessing where masks were; its accuracy would then track the masking rate, not understanding of the text.</details>
2. Why does `replace_tokens` receive `gen_logits.data` and not the Tensor?
   <details><summary>Answer</summary>The sampled ids are integers chosen by a non-differentiable inverse CDF. Passing plain data makes explicit that no gradient flows from $L_D$ into the generator; the generator then learns only from $L_G$, as the paper specifies.</details>
3. What goes wrong if the discriminator reads `inputs` instead of `corrupt`?
   <details><summary>Answer</summary>`inputs` still contains [MASK] at 80% of picked positions. The discriminator could score well by spotting [MASK] tokens, which never appear downstream, and would not learn to judge plausible replacements.</details>
4. Why is $\lambda = 50$ on $L_D$ and not on $L_G$?
   <details><summary>Answer</summary>$L_D$ is a binary cross-entropy averaged over all real tokens, much smaller per token than the $V$-way $L_G$. Weighting $L_D$ brings the discriminator's gradient up to a useful scale; putting the 50 on $L_G$ would let the generator dominate the shared embeddings and multiply its gradients by 50.</details>
5. In the worked example, why does the step use exactly two uniforms after masking?
   <details><summary>Answer</summary>`replace_tokens` draws one uniform per position whose label is not $-100$, and only positions 1 and 3 were picked. Unpicked and padded positions draw nothing.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. every masked position labelled replaced | the discriminator learns where the masks were; its accuracy tracks the masking rate | `is_replaced = (corrupt != x)` |
| 2. a uniform drawn at every position | a seed gives another corruption than the spec, so Python and a replay disagree | `replace_tokens` skips positions whose label is `ignore_index` before drawing |
| 3. a head without the GELU | logits differ from HF; pretrained checkpoints score differently | `dense_prediction(F.gelu(dense(h)))` |
| 4. padding in the loss or in the masking | short sequences in a batch weigh less; padding gets replaced | `rtd_loss` indexes by `real` before the mean; `mlm_mask` gets `special \| ~real` |
| 5. the discriminator reading the masked inputs | it sees `[MASK]` where the replacement should be: the task becomes "find the masks" | `disc(corrupt, ...)` |
| 6. $\lambda$ on the wrong loss | the generator's MLM loss dominates and its gradients grow 50 times | `gen_loss + disc_loss * lam` |
| 7. untied embeddings | two embedding tables, the generator's trained only by $L_G$ | the generator's three embedding weights are the discriminator's Tensors |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L6.2` | `mlm_mask` picks positions; `BertForMLM` is the generator and `BertEncoder` the discriminator's body |
| Back | `M07.1` | `sample_categorical` draws the replacements |
| Back | `L0.3` | `bce_with_logits` is the replaced-token loss |
| Back | `L0.1` | the Tensor and `no_grad` (the zoo scores without a graph) |
| Back | `L0.2` | `gelu` and `reshape` in the head |
| Back | `L0.4` | the head's `Linear` layers and the `Module` registration order |
| Back | `L0.6` | the model directory's safetensors |
| Back | `M06.3` | PCG32: the default init stream and every draw of the step |
| Back | `M07.3` | `normal_init` for the head's weights |
| Forward | `L6.7` | the zoo loads ELECTRA directories and reports replaced-token detection accuracy next to BERT's masked-token accuracy |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `ElectraDiscriminator` | HF `ElectraForPreTraining` | `embeddings_project` when the embedding size differs from the hidden size | `transformers/models/electra/modeling_electra.py` |
| `electra_step` | the original ELECTRA pretraining code | a generator a third to a quarter of the discriminator's width; gumbel-noise sampling; masking 85% [MASK], 15% kept | `google-research/electra`, `pretrain/pretrain_helpers.py` |
| replaced-token detection | DeBERTaV3 | gradient-disentangled embedding sharing: the generator's gradients stop at the shared embeddings | He, Gao, and Chen, "DeBERTaV3" (2021) |
| ELECTRA as a backbone | ELECTRA-small fine-tuned on GLUE | SST-2 and the other GLUE tasks with a classification head | `L6.5`; the paper's table 1 |
