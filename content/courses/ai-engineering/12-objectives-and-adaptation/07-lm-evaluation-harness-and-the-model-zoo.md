# The evaluation harness and the model zoo

Ten kinds of model have been trained by now: a [byte bigram](../06-numerics-and-autograd/06-training-loop-and-the-autograd-bigram.md), a [Kneser-Ney n-gram](../08-statistical-language-models/01-ngram-kneser-ney.md), a [neural probabilistic LM](../08-statistical-language-models/02-bengio-nplm.md), [word vectors](../08-statistical-language-models/03-word2vec-and-ppmi-svd.md), an [RNN language model](../09-recurrent-networks/06-rnn-language-model-stateful-tbptt.md), an [attention seq2seq](../10-attention/01-encoder-decoder-teacher-forcing.md), a [Transformer](../11-transformer/05-encoder-decoder-transformer.md), [GPT](01-gpt-decoder-only.md), [BERT](02-bert-masked-lm.md), and [ELECTRA](03-electra-replaced-token-detection.md), each with its own little evaluation in its own section. They cannot be compared. The n-gram reports perplexity per byte, GPT per window, the classifier an accuracy on whatever split was handy, and none says how sure it is.

The [TinyStories capstone](../capstones/01-tinystories.md) has to put a 10M-parameter Llama in a table next to all of them, and the durable [`EvalSuite` workflow](../../systems/07-durable-orchestration/11-trainrun-and-evalsuite.md) in *Systems Engineering* has to rerun that table on every release. This section writes the measuring instrument once: one definition of perplexity for fixed-context models, one of multiple-choice accuracy, an interval on every number, a paired test for "better", and a zoo that loads each family through the checkpoint it already saves. By the end, `eval ppl` reports a perplexity with an interval, and `eval --suite zoo` reports one row per family.

The statistics come from *Mathematics*: the [NLL accumulator, perplexity, and bits per byte](../../mathematics/17-information-theory/02-perplexity-bits-per-byte-nll-accumulator.md), [confidence intervals and the bootstrap](../../mathematics/13-probability-statistics/05-lln-clt-confidence-intervals-bootstrap.md), and the [paired permutation test](../../mathematics/13-probability-statistics/06-hypothesis-tests.md). The zoo also uses [beam search](../10-attention/04-beam-search.md), [exact match](../10-attention/05-sequence-metrics.md), and the [classifiers](05-fine-tuning-heads-and-the-linear-head-export.md) of this chapter. The reference implementation is two files: [`python/tinyllm/eval/lm.py`](../../../../course/solutions/python/tinyllm/eval/lm.py) (`Result`, `ByteTokenizer`, `log_probs`, `token_nlls`, `eval_ppl`, `score_choices`, `run_task`, `compare`) and [`python/tinyllm/eval/zoo.py`](../../../../course/solutions/python/tinyllm/eval/zoo.py) (`read_config`, `kind_of`, `save_word2vec`, `load_model`, `lm_token_nlls`, `encode_batch`, `score_entry`, `run_zoo`, `write_report`). Their contracts are [`lm.pyi`](../../../../course/contracts/py/tinyllm/eval/lm.pyi) and [`zoo.pyi`](../../../../course/contracts/py/tinyllm/eval/zoo.pyi), and the report format is [`formats/eval-results.schema.json`](../../../../course/contracts/formats/eval-results.schema.json).

> **Learning objectives**
> - Compute strided perplexity so that every token is scored exactly once with at least `ctx_len - stride` tokens of context.
> - Score multiple-choice items by log-likelihood, and explain when `acc` and `acc_norm` disagree.
> - Choose the right 95% interval for each metric: Student t, bootstrap, Wilson, or Fisher z.
> - Compare two models with a paired sign-flip test rather than overlapping intervals.
> - Build a zoo that scores every family in its own unit and turns failures into rows.

## Strided perplexity

Here is the notation for this section.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x_0, \dots, x_{n-1}$ | the evaluation tokens | `int64[n]` |
| $\ell_t = -\ln p(x_t \mid \text{context})$ | the NLL of token $t$, in nats | `float64` |
| $C$, $s$ | `ctx_len` and `stride`, $1 \le s < C$ | `int` |
| $[a_k, e_k)$ | window $k$: $a_k = k s$, $e_k = \min(a_k + C, n)$ | `int` |
| $\bar\ell = \frac{1}{N} \sum_t \ell_t$ | mean NLL over the $N$ scored tokens | `float` |
| $\mathrm{ppl} = e^{\bar\ell}$ | perplexity | `float` |
| $\mathrm{bpb} = \sum_t \ell_t / (n_{\text{bytes}} \ln 2)$ | bits per byte | `float` |
| $\mathrm{LL}(c) = \sum_j \ln p(c_j \mid \text{ctx}, c_{<j})$ | a choice's log-likelihood | `float` |
| $a_i, b_i$ | two models' scores on item $i$ (`per_item`) | `float64[n]` |

A model with a fixed context of $C$ tokens cannot read a long text at once. Cutting the text into disjoint blocks of $C$ is cheap but unfair: the first tokens of every block are predicted from almost nothing. Sliding the window one token at a time is fair but costs $n$ forward passes. The strided compromise moves the window by $s$. Window $k$ covers $[a_k, e_k)$ and scores only the tokens it **adds**, $t \in [\max(e_{k-1}, 1), e_k)$, each from its own log-probabilities at position $t - 1 - a_k$. The loop stops at the first window that reaches $n$.

Three facts follow, and a correct solution must guarantee each:

1. Every token $1, \dots, n - 1$ is scored exactly once (token 0 has no context).
2. In every window after the first, a scored token has at least $C - s$ tokens of context.
3. $s = C$ is not allowed. Each window's first token would be predicted with no context from its own window, which is the bug in many copies of the popular Hugging Face recipe. $s = C - 1$ is the cheapest valid stride and $s = 1$ the fairest.

The totals go through the `NLLAccumulator`: compensated sums, then $\bar\ell$, $\mathrm{ppl}$, and, given the byte count, $\mathrm{bpb}$. Only bits per byte compares models with different tokenizers; for the book's byte-level models one token is one byte.

## Multiple choice by log-likelihood

Perplexity measures a model on running text. Tasks measure it on questions. A task item is a context and $k$ choices, and each choice is scored by its log-likelihood after the context: run the model on context plus choice and sum the log-probabilities of the choice's tokens, each read at the position before it. `acc` takes $\arg\max_c \mathrm{LL}(c)$. `acc_norm` takes $\arg\max_c \mathrm{LL}(c) / \mathrm{bytes}(c)$, because a sum of negative numbers favors short answers. Ties go to the first choice. This is the lm-evaluation-harness recipe, and the same function scores the capstone's held-out tasks.

## Every metric with an interval

A metric computed from $n$ items is an estimate, so the zoo reports its 95% interval. Different statistics call for different intervals:

| metric | interval | why that one |
|---|---|---|
| mean NLL, bpb, ppl | Student t on the per-token NLLs; ppl's ends are $e^{\text{ends}}$ | NLLs are many and roughly independent within a text |
| task accuracy (`run_task`) | percentile bootstrap of the per-item scores from the given rng | works for any statistic; reproducible in Go ([paired A/B experiments](../29-evaluation/04-ab-experiments.md)) |
| zoo accuracy and EM | Wilson score interval | correct coverage for small $n$ and accuracies near 0 or 1 |
| Spearman | Fisher z: $\tanh(\operatorname{atanh}\rho \pm 1.96 / \sqrt{n - 3})$ | the standard interval for a rank correlation |

## Paired comparison

An interval says how sure one number is. "Is B better than A?" is a different question. Two models scored on the same items are not two independent samples, because the items' difficulty is shared. `compare` runs the paired permutation (sign-flip) test on $d_i = a_i - b_i$: under "no difference", each $d_i$ is as likely to have either sign, so flipping signs at random gives the null distribution of $\lvert\sum_i d_i\rvert$. A small p-value says the difference is not item noise. Comparing the two means with unpaired intervals throws the pairing away and misses real differences.

## The model zoo

With the measurements defined, the zoo applies them to every family. `load_model(dir)` reads `config.json` (the n-gram's safetensors file names its arch in its metadata instead) and dispatches on `tl_arch` to the family's own loader; a config with `tl_head` is a fine-tuned classifier. `run_zoo(manifest, rng)` scores each manifest entry according to its kind:

| kind | families | task | metric |
|---|---|---|---|
| lm | bigram, ngram, nplm, rnnlm, gpt | a token file ([`formats/tokens-bin.md`](../../../../course/contracts/formats/tokens-bin.md)) | bpb from the family's own per-token NLLs (gpt: strided) |
| seq2seq | seq2seq, transformer | JSONL `{"src", "tgt"}` | exact match of the best beam hypothesis without its eos |
| classifier | any `tl_head` | `split, sentence, label` rows | accuracy on the split |
| mlm | bert | the same rows | accuracy of the masked tokens |
| rtd | electra | the same rows | replaced-token detection accuracy |
| embeddings | word2vec | `w1, w2, score` lines | Spearman |

The report follows [`formats/eval-results.schema.json`](../../../../course/contracts/formats/eval-results.schema.json), with one row per entry in manifest order. A row that cannot be scored is not dropped. An exception becomes `status: "error"` with its type and message, and an unknown family becomes `status: "skipped"`, both with `value: null`, and the rest of the zoo is still scored. A table that silently loses its broken rows looks better than the system it measures.

## Working it by hand

Start with **strided windows**. Take six tokens, $C = 4$, and a model that gives the true next token probability $1 / (j + 2)$ at window position $j$ (it has read the $j + 1$ tokens up to there), so each NLL reveals the context it was scored with. The table compares two strides.

| stride | windows | tokens scored by each | NLLs |
|---|---|---|---|
| 2 | $[0, 4)$, $[2, 6)$ | 1, 2, 3 at $j = 0, 1, 2$; then 4, 5 at $j = 1, 2$ | $\ln 2, \ln 3, \ln 4, \ln 3, \ln 4$ |
| 3 | $[0, 4)$, $[3, 6)$ | 1, 2, 3; then 4, 5 at $j = 0, 1$ | $\ln 2, \ln 3, \ln 4, \ln 2, \ln 3$ |

With stride 2 the mean is $(\ln 2 + 2 \ln 3 + 2 \ln 4) / 5$, so $\mathrm{ppl} = (2 \cdot 3^2 \cdot 4^2)^{1/5} = 288^{1/5} = 3.1037$. The larger stride is cheaper and gives tokens 4 and 5 less context.

Next, **multiple choice**. The context is "a", the choices are "bb" and "c", and $p(b \mid a) = 0.4$, $p(b \mid b) = 0.9$, $p(c \mid a) = 0.45$.

| choice | LL | bytes | LL per byte |
|---|---|---|---|
| "bb" | $\ln 0.4 + \ln 0.9 = -1.0217$ | 2 | $-0.5108$ |
| "c" | $\ln 0.45 = -0.7985$ | 1 | $-0.7985$ |

`acc` picks "c", and `acc_norm` picks "bb".

Finally, **the zoo's unit**. A bigram whose table is all zeros gives every byte probability $1/256$, so $\ell_t = \ln 256$ and $\mathrm{bpb} = \ln 256 / \ln 2 = 8$ exactly, with a zero-width interval because every NLL is the same.

## Designing the interface

```python
@dataclass
class Result: value: float; lo: float; hi: float; n: int; per_item: NDArray
def token_nlls(model, ids, ctx_len: int, stride: int) -> NDArray: ...          # [n - 1]
def eval_ppl(model, ids, ctx_len, stride, n_bytes=0) -> dict[str, float]: ...
def score_choices(model, tok, context, choices, normalize=True, max_len=None) -> NDArray: ...
def run_task(model, tok, task_jsonl, metric, rng, n_boot=1000, max_len=None) -> Result: ...
def compare(a: Result, b: Result, n_perm=10000, rng=None) -> float: ...
def load_model(dir) -> Any: ...                     # zoo dispatch on tl_arch
def run_zoo(manifest, rng, seed=0) -> dict: ...     # formats/eval-results report
```

A model is anything callable as `model(ids[None])` that returns logits `[1, T, V]`, or a tuple whose first item is the logits (GPT returns `(logits, loss)`). `Result` keeps `per_item` so that two results can be paired later.

## Building it step by step

The reasoning runs as follows.

1. Perplexity is $e^{\bar\ell}$ with $\bar\ell$ the mean NLL of the tokens, so the harness must produce one NLL per token $1..n-1$, each conditioned on as much context as the model allows, and none twice.
2. A fixed-context model sees at most $C$ tokens. Sliding by $s < C$ and scoring only the tokens a window **adds** gives every scored token at least $C - s$ tokens of context and scores each exactly once. $s = C$ would score each window's first token with no in-window context, so it is refused.
3. Inside window $[a, e)$, the logits at position $j$ predict token $a + j + 1$, so token $t$'s log-probability is read at row $t - 1 - a$. This one index is where most harnesses go wrong.
4. Multiple choice reduces to the same quantity: the log-likelihood of the choice's tokens after the context, read at the position before each token. Normalizing by bytes removes the length bias of summed log-probabilities, and ties go to the first choice so results do not depend on the argmax implementation.
5. A metric is an estimate from finitely many items, so each carries a 95% interval chosen for its statistic. Comparing models is a paired question, so `compare` uses the sign-flip test on per-item differences.
6. For a zoo of heterogeneous families the comparison unit is per kind: bits per byte for LMs (the only unit fair across tokenizers), exact match for seq2seq, accuracy for classifiers and pretraining objectives, and Spearman for word vectors. Each family is loaded through its own saved directory, dispatched on `tl_arch`.
7. A report that drops broken rows overstates the system. Every entry produces a row: `ok`, `error` with the exception, or `skipped` for an unknown family.

### Log-probabilities from any model

One adapter makes every family look the same to the harness:

```python
def log_probs(model, ids):
    x = np.asarray(ids, dtype=np.int64)
    with no_grad():                       # scoring builds no graph
        out = model(x[None])
    if isinstance(out, tuple):  # GPT returns (logits, loss)
        out = out[0]
    z = np.asarray(out.data if isinstance(out, Tensor) else out, dtype=np.float64)[0]
    # ... one row of logits per id ...
    z = z - z.max(axis=-1, keepdims=True)                 # stable log-softmax in float64
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
```

### Strided NLLs

```python
def token_nlls(model, ids, ctx_len, stride):
    x = np.asarray(ids, dtype=np.int64)
    n = x.size
    # ... 1-D, at least 2 tokens ...
    if not 1 <= stride < ctx_len:
        # stride == ctx_len would leave each window's first token unscored.
        raise ValueError(...)
    out = np.empty(n - 1, dtype=np.float64)   # one NLL per token 1..n-1
    prev = 0                                  # end of the previous window
    for s in range(0, n, stride):
        e = min(s + ctx_len, n)
        lp = log_probs(model, x[s:e])
        # Only the tokens this window adds; the overlap was scored by an
        # earlier window. Each new token sees at least ctx_len - stride
        # tokens of context (all of them near the start).
        for t in range(max(prev, 1), e):
            out[t - 1] = -lp[t - 1 - s, x[t]]   # row t-1-s predicts token t
        prev = e
        if e == n:                               # the first window to reach the end is the last
            break
    return out
```

In the worked example with $C = 4$ and $s = 2$, the first window $[0, 4)$ scores tokens 1 to 3 at rows 0 to 2. The second, $[2, 6)$, starts its scoring at `prev` $= 4$, so it scores tokens 4 and 5 at rows 1 and 2. Without the `break`, a third window $[4, 6)$ would run for nothing.

### Perplexity with an interval

```python
def eval_ppl(model, ids, ctx_len, stride, n_bytes=0):
    nll = token_nlls(model, ids, ctx_len, stride)
    acc = NLLAccumulator()  # compensated sums, bpb
    acc.add(nll, n_bytes=int(n_bytes))
    r = acc.result()
    out = {k: float(r[k]) for k in ("nll_sum", "n_tokens", "nll_mean", "ppl", "bits_per_token")}
    if n_bytes:
        out["bpb"] = float(r["bpb"])
    if nll.size >= 2:
        _, lo, hi = mean_ci(nll)  # Student t over the per-token NLLs
        # The interval is on the mean NLL; exp is monotone, so its ends map to ppl's ends.
        out["ppl_lo"], out["ppl_hi"] = math.exp(lo), math.exp(hi)
        if n_bytes:
            scale = nll.size / (n_bytes * math.log(2.0))   # mean nats/token -> bits/byte
            out["bpb_lo"], out["bpb_hi"] = lo * scale, hi * scale
    return out
```

### Choices and tasks

```python
def score_choices(model, tok, context, choices, normalize=True, max_len=None):
    ctx = list(tok.encode(context))
    # ... non-empty context ...
    scores = np.empty(len(choices), dtype=np.float64)
    for i, ch in enumerate(choices):
        cont = list(tok.encode(ch))
        # ... non-empty choice ...
        seq = ctx + cont
        if max_len is not None and len(seq) > max_len:
            seq = seq[-max_len:]  # keep the end: the choice and the nearest context
            # ... at least one context token must remain ...
        lp = log_probs(model, seq)
        k = len(seq) - len(cont)                  # index of the choice's first token
        # Token k + j is predicted by row k + j - 1.
        ll = float(sum(lp[k - 1 + j, cont[j]] for j in range(len(cont))))
        scores[i] = ll / len(ch.encode("utf-8")) if normalize else ll   # acc_norm: per byte
    return scores
```

```python
    for i, it in enumerate(items):
        s = score_choices(model, tok, it["context"], it["choices"], normalize=metric == "acc_norm", max_len=max_len)
        right[i] = float(int(np.argmax(s)) == int(it["label"]))  # ties to the first choice
    value, lo, hi = bootstrap_ci(right, np.mean, n_boot, 0.05, rng)
    return Result(float(value), float(lo), float(hi), len(items), right)
```

`Result.per_item` keeps the 0/1 scores so that `compare` can pair them:

```python
def compare(a, b, n_perm=10000, rng=None):
    if a.per_item.shape != b.per_item.shape:
        raise ValueError(...)        # pairing needs the same items
    r = rng if rng is not None else PCG32(0).substream("sample")
    return paired_permutation_test(a.per_item, b.per_item, n_perm, r)
```

### The zoo: dispatch

One table names each family's kind and metric:

```python
# tl_arch -> (kind, metric, higher_is_better)
FAMILIES = {
    "bigram": ("lm", "bpb", False), "ngram": ("lm", "bpb", False), "nplm": ("lm", "bpb", False),
    "rnnlm": ("lm", "bpb", False), "gpt": ("lm", "bpb", False),
    "seq2seq": ("seq2seq", "em", True), "transformer": ("seq2seq", "em", True),
    "bert": ("mlm", "accuracy", True), "electra": ("rtd", "accuracy", True),
    "word2vec": ("embeddings", "spearman", True),
}

def kind_of(cfg):
    if cfg.get("tl_head"):
        return "classifier"  # a fine-tuned head over any backbone
    arch = cfg.get("tl_arch")
    if arch not in FAMILIES:
        raise KeyError(f"no zoo scorer for tl_arch {arch!r}")   # becomes a "skipped" row
    return FAMILIES[arch][0]
```

`read_config` reads `config.json`, or, for the n-gram (which saves one safetensors file), the arch named in its metadata. `load_model` calls the family's own loader (`load_gpt`, `load_bert`, `load_electra`, `load_transformer`, and so on).

### The zoo: LM rows in bits per byte

Each family supplies per-token NLLs its own way. The bigram is a log-softmax of its table, read `[previous, next]`:

```python
def lm_token_nlls(model, arch, cfg, ids, entry):
    x = np.asarray(ids, dtype=np.int64)
    if arch == "bigram":
        W = np.asarray(model.weight, dtype=np.float64)
        z = W - W.max(axis=1, keepdims=True)
        lsm = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -lsm[x[:-1], x[1:]]            # row = previous byte, column = next byte
    if arch == "ngram":
        return np.asarray(model.nll(x.tolist()), dtype=np.float64)
    if arch in ("nplm", "rnnlm"):
        return np.asarray(model.nll(x), dtype=np.float64)
    if arch == "gpt":
        ctx = int(entry.get("ctx_len", model.cfg.n_ctx))
        return token_nlls(model, x, ctx, int(entry.get("stride", max(1, ctx // 2))))
    raise KeyError(...)
```

```python
    if kind == "lm":
        ids = np.asarray(open_tokens(str(b / entry["data"])), dtype=np.int64)
        nll = lm_token_nlls(model, arch, cfg, ids, entry)
        # ... byte models: one scored token is one byte; otherwise the entry gives n_bytes ...
        _, lo, hi = mean_ci(nll)  # Student t over per-token NLLs
        scale = nll.size / (n_bytes * math.log(2.0))
        row.update(metric="bpb", value=float(nll.sum() / (n_bytes * math.log(2.0))),
                   ci95=[lo * scale, hi * scale], n=int(nll.size))
```

A uniform bigram (all-zero table) gives every NLL $\ln 256$, so the value is exactly 8, and `mean_ci` of a constant gives a zero-width interval.

### The zoo: the other kinds

Seq2seq rows decode with `beam_search` (or `translate` for the transformer), strip eos from finished hypotheses, and score exact match with a Wilson interval. Classifier, MLM, and RTD rows read only the requested split (default `val`) of a `split, sentence, label` file, encode bytes between `[CLS]` and `[SEP]`, and count correct predictions, masked tokens recovered, or correct replaced/original calls, again with a Wilson interval. The word2vec row computes Spearman with `word_similarity` and a Fisher z interval with standard error $1/\sqrt{n - 3}$, where $n$ is the number of pairs whose words are both in the vocabulary.

### The zoo: failures are rows

```python
def run_zoo(manifest, rng, seed=0):
    m = json.loads(Path(manifest).read_text())
    base = str(Path(manifest).resolve().parent)    # entry paths are relative to the manifest
    rows = []
    for entry in m["models"]:
        try:
            rows.append(score_entry(entry, base, rng))       # under no_grad
        except KeyError as e:
            if "no zoo scorer" in str(e):
                rows.append({..., "status": "skipped", "value": None, "reason": ...})
                continue
            rows.append(_error_row(entry, e))
        except Exception as e:  # noqa: BLE001 - one broken family must not hide the rest
            rows.append(_error_row(entry, e))
    return {"format": FORMAT, "suite": m.get("suite", "zoo"), "seed": int(seed), "manifest": manifest, "rows": rows}
```

The solution's entry point, [`cli_obj.py`](../../../../course/solutions/entry/python/tinyllm/cli_obj.py), exposes these as `eval ppl --model --data --ctx --stride`, `zoo add` (which appends a manifest entry), and `eval --suite zoo --manifest --out`, the command `EvalSuite` runs. The full solution is in [`lm.py`](../../../../course/solutions/python/tinyllm/eval/lm.py) and [`zoo.py`](../../../../course/solutions/python/tinyllm/eval/zoo.py).

### Why this design

Strided perplexity runs $\lceil (n - C)/s \rceil + 1$ windows of at most $C$ tokens: $s = C - 1$ is about $n / C$ forward passes, and $s = 1$ about $n$. Multiple choice is one forward pass per choice.

The harness **scores only what a window adds** rather than averaging overlapping scores. Each token's NLL comes from the first window that reaches it, which (for $t \ge C$) is the one giving it at least $C - s$ context; averaging would mix in worse-conditioned predictions. It compares families in **bits per byte** rather than perplexity, because perplexity per token depends on the tokenizer and bits per byte does not. It uses **per-metric intervals** rather than one bootstrap for everything: Student t and Wilson are exact enough and cheap, and the bootstrap is used where the statistic (task accuracy) is reproduced in Go. It turns **failures into rows** rather than exceptions, because `EvalSuite` must always get a full table to compare releases. The tradeoff accepted is that the zoo's per-family adapters are code the harness must maintain as families change; but each uses that family's own saved format, which is the point of the zoo.

## Testing it

A correct solution has to pass these checks.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand-worked strided windows | unit | the windows the model is called on, the NLLs, $288^{1/5}$ | everyone agrees on the definition |
| hand-worked choices | unit | both sums, both normalizations, the two winners | acc and acc_norm |
| a uniform bigram is 8 bits per byte | unit | a uniform byte model scores exactly 8 bpb, zero-width interval | the unit every LM row is in |
| agreement with Hugging Face GPT-2 | golden | per-token NLLs of HF's GPT-2 under four (ctx, stride) settings, loaded by `load_hf_gpt2` | the definition on a real model |
| every token scored once | property | 25 random settings: $n - 1$ tokens, context bounds; stride $\ge$ ctx refused | no token skipped or counted twice |
| ppl, bpb and their interval | differential | the accumulator's totals and the Student t interval, rescaled to ppl and bpb | numbers with error bars |
| task accuracy with bootstrap | differential | per-item scores, ties to the first choice, the bootstrap from the same rng | task tables in the capstone and quantization |
| `compare` is the paired permutation test | differential | the paired test's p-value on the per-item scores; mismatched items refused | "better" means significant on the same items |
| LM rows use each family's own NLL | differential | five LM families: bpb and interval from each family's own NLL | the capstone's baseline table |
| seq2seq rows decode with beam | differential | EM of the beam decode without eos; transformer through `translate` | the seq2seq and transformer rows |
| accuracy rows | unit | classifier, BERT, and ELECTRA rows with Wilson intervals; the val split only | this chapter's backbone comparison |
| the word-vector row | unit | Spearman with the Fisher z interval | the word2vec row |
| the report validates against the schema | conformance | [`eval-results.schema.json`](../../../../course/contracts/formats/eval-results.schema.json), row order, seed | `EvalSuite` reads it |
| failures are rows | fault | a missing directory is an error row, an unknown family a skipped row, the healthy row is scored | one broken family never hides the rest |

The model tests in [`test_eval.py`](../../../../course/solutions/learner-tests/L6.7/python/tests/l6-7-eval/test_eval.py) use a toy model whose NLL encodes its context length, log-likelihoods from a table, the interval and test functions called directly, and checkpoints whose score is known exactly.

## What can go wrong

Evaluation bugs are dangerous because they produce plausible numbers. The common ones are these.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| rescoring the overlap, or allowing stride = ctx_len | tokens scored with less context than they had; window starts predicted from nothing | the inner loop starts at `max(prev, 1)`; `1 <= stride < ctx_len` is enforced |
| reading a choice's log-probabilities one position late | the score of the wrong tokens; accuracy near chance | `lp[k - 1 + j, cont[j]]` |
| acc and acc_norm swapped, or ties to the last choice | short-answer bias where none was asked for; results that depend on choice order | `normalize = metric == "acc_norm"`; `np.argmax` returns the first maximum |
| bits per byte in nats, or the bigram table read transposed | LM rows off by $\ln 2$; a bigram scored on the reversed text | divides by `n_bytes * log(2)`; reads `lsm[x[:-1], x[1:]]` |
| intervals left in log space, or Fisher z with $\sqrt n$ | a perplexity interval in nats; Spearman intervals too narrow | `exp` of the NLL interval's ends; `se = 1 / sqrt(n - 3)` |
| comparing unpaired items | a real difference reported as noise, or the reverse | `compare` requires equal `per_item` shapes and runs the paired test |
| exact match with the eos still on | every seq2seq row scores 0 | `best.tokens[:-1] if best.finished`; `translate` strips eos too |
| classifier rows scored on every split | training sentences inflate the accuracy | `_text_rows` keeps only `entry.get("split", "val")` |

> **Warning:** The off-by-one between logit row and target token is the single most common evaluation bug. Row $j$ of a window predicts token $j + 1$. If accuracy on a multiple-choice task is near chance for a model that generates well, check this index first.

## Where this leads

The first caller is [quantization](../21-inference/05-quantization.md), whose `quant_ppl` runs `eval_ppl` on a float model and its quantized copy to measure the perplexity cost Part 8 budgets. The [Llama-family model](../13-decoder-block/09-llama-family-model.md) reports SmolLM2 parity in bits per byte through `eval_ppl`. The [TinyStories capstone](../capstones/01-tinystories.md)'s baseline table is `run_zoo` over every family trained so far, and [supervised fine-tuning](../18-post-training/01-supervised-fine-tuning.md) evaluates with the same harness. In *AI Engineering*, the [`EvalSuite` workflow](../../systems/07-durable-orchestration/11-trainrun-and-evalsuite.md) execs `{tinyllm} eval --suite zoo` and keeps the report.

The production tools scale the same ideas. Hugging Face's "Perplexity of fixed-length models" recipe batches windows on GPU (and carries an off-by-one in the token count you no longer have). EleutherAI's lm-evaluation-harness (`lm_eval/api/model.py`, `lm_eval/api/task.py`) adds hundreds of tasks, few-shot prompts, `loglikelihood_rolling`, caching, and request reordering. HELM and the harness's stderr add per-task standard errors and a paired bootstrap across scenarios. And the Open LLM Leaderboard and OpenCompass run many models on many tasks, with result stores and versioned task definitions.

Further reading:

- Hugging Face, "Perplexity of fixed-length models" (`transformers` docs, `perplexity.md`).
- Gao et al., EleutherAI lm-evaluation-harness (2021 onward).
- Biderman et al., "Lessons from the Trenches on Reproducible Evaluation of Language Models" (2024).
- Dror et al., "The Hitchhiker's Guide to Testing Statistical Significance in NLP" (2018).
- Liang et al., "Holistic Evaluation of Language Models" (2022).

## Key terms

- **Strided perplexity**: perplexity computed over overlapping windows that each score only the tokens they add.
- **Bits per byte**: total NLL in bits divided by the text's bytes, the tokenizer-independent unit of the zoo.
- **`acc_norm`**: multiple-choice accuracy using log-likelihood per byte, removing the length bias of summed log-probabilities.
- **Paired permutation test**: a sign-flip test on per-item score differences between two models scored on the same items.
- **Model zoo**: the manifest-driven report that scores every model family in its own unit, one row per entry.

## Summary

- Strided perplexity scores every token exactly once, each from the first window that reaches it, with at least `ctx_len - stride` tokens of context.
- Multiple choice is the log-likelihood of each answer after the context; `acc_norm` divides by the answer's bytes, which can flip the winner.
- Every metric carries a 95% interval: Student t for mean NLL, bootstrap for task accuracy, Wilson for proportions, Fisher z for Spearman.
- "Is B better than A?" is a **paired** question on the same items, answered by the sign-flip test.
- The zoo loads every family through its own checkpoint loader and scores it on its own task in one report; a broken entry becomes a row with a reason.

## Review questions

1. Why does `token_nlls` read `lp[t - 1 - s, x[t]]` and not `lp[t - s, x[t]]`?
   <details><summary>Answer</summary>The logits at window position $j$ are the prediction for the next token, $a + j + 1$. Token $t$ is predicted at $j = t - 1 - a$. Reading at $t - a$ would score each token with the prediction meant for its successor.</details>
2. Why is `stride == ctx_len` refused?
   <details><summary>Answer</summary>Windows would not overlap, so each window's first new token sits at position 0 of its window and would need the previous window's last token as context, which the window does not contain. Those tokens would be predicted from nothing (or skipped), biasing perplexity upward.</details>
3. Why can `acc` and `acc_norm` pick different answers, as in the worked example?
   <details><summary>Answer</summary>A log-likelihood is a sum of negative terms, so longer answers have lower sums. "bb" has a better per-byte score ($-0.51$) than "c" ($-0.80$) but a worse total ($-1.02$ vs $-0.80$). Dividing by bytes removes the length bias.</details>
4. Why is `compare` a paired test rather than a check that the two intervals overlap?
   <details><summary>Answer</summary>Both models are scored on the same items, so item difficulty is shared and cancels in $a_i - b_i$. Overlapping unpaired intervals ignore that and can hide a consistent per-item advantage.</details>
5. Why does an unknown `tl_arch` produce a `skipped` row rather than an `error` row?
   <details><summary>Answer</summary>It is not a failure of a checkpoint but a family the zoo has no scorer for yet (for example a future `llama` row). Marking it skipped keeps it visible in the table without counting it as a broken model.</details>
