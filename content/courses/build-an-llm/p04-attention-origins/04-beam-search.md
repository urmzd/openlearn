# Beam search, generic over a step function

## Overview

| | |
|---|---|
| **Module** | `L4.4` · build · Python · Pass 4 · 1.5 to 2 h reading |
| **Solution** | [`python/tinyllm/infer/beam.py`](../../../../course/solutions/python/tinyllm/infer/beam.py): `Hypothesis`, `select_state`, `beam_search`, `greedy_decode`; model oracle tests in [`test_beam.py`](../../../../course/solutions/learner-tests/L4.4/python/tests/l4-4-beam/test_beam.py) |
| **Contract** | [`course/contracts/py/tinyllm/infer/beam.pyi`](../../../../course/contracts/py/tinyllm/infer/beam.pyi) |
| **Needs** | `M09.2` `log_softmax` · `L4.1` `Seq2Seq.decode_step` and `L4.3` `LuongAttention` (a real decoder to search over) · reading: `S-M06a` (counting sequences) |
| **Used by** | later: `L6.7` the zoo's beam decoding of seq2seq and transformer checkpoints (joins the registry with B7) · later: `L5.5` · `{tinyllm} translate --beam 5` runs it, and on the dates task it beats greedy |
| **Optional depth** | Graves, "Sequence Transduction with Recurrent Neural Networks" (2012), section 3.2; Wu et al., "Google's Neural Machine Translation System" (2016), section 7 (length penalty); Meister, Cotterell, and Vieira, "If beam search is the answer, what was the question?" (EMNLP 2020) |

## Key Takeaways

- Greedy decoding commits to the best next token and can miss the best sequence; beam search keeps the `beam_size` best prefixes and finds it in the worked example, 0.36 against greedy's 0.35.
- All `k * V` extensions are ranked **together**, so the beam never holds more than `beam_size` rows.
- A beam of 1 is greedy decoding, and a beam of at least $V^L$ is exact search; both are checked against independent oracles.
- The model is only a step function: its state is gathered by parent with `select_state`, so the same search decodes a toy table and a real `Seq2Seq`.
- Summed log-probability prefers short outputs; dividing by `len ** length_penalty` can change the winner.

## How to read this chapter

Sections 1 to 3 build beam search from first principles and work a case where it beats greedy by hand. Section 4 fixes the interface and what a correct solution must handle. Section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

The seq2seq model of `L4.1` can translate, but only greedily: at each step it takes the single most likely token and never looks back. On the reversal task that is fine, because the model is nearly certain at every step. On the dates task, where the first digit of the year is often uncertain, a greedy decoder commits to a likely-looking first token and then has to live with a worse whole output. This module builds the search that fixes it, once, as a function of a step function, so that later the zoo (`L6.7`) uses the same code for seq2seq and transformer checkpoints. The model and the search are separate pieces with a narrow interface between them, which is how production decoders (vLLM's beam search, Hugging Face's `generate`) are built too.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $V$ | vocabulary size | `int` |
| $y_{1:t}$ | a prefix: the tokens generated after `bos` | list of ids |
| $p(v \mid y_{1:t})$ | the model's next-token distribution, softmax of its logits | `float64[V]` |
| $\ell(y_{1:t}) = \sum_{i \le t} \log p(y_i \mid y_{1:i-1})$ | the prefix's log-probability (`logprob`) | `float` |
| $K$ | `beam_size` | `int` |
| $k$ | number of live hypotheses at this step, $k \le K$ | `int` |
| $L$ | `max_len`, the most tokens generated | `int` |
| $\alpha$ | `length_penalty` | `float` $\ge 0$ |
| $s(y) = \ell(y) / \lvert y \rvert^\alpha$ | the score a finished hypothesis is ranked by | `float` |

### 2.1 The search problem

A decoder model defines $P(y) = \prod_i p(y_i \mid y_{1:i-1})$ over sequences that end with `eos`. The best output is $\arg\max_y \ell(y)$. There are $V^L$ sequences of length $L$; with $V = 32\,000$ and $L = 20$ exhaustive search is impossible, and dynamic programming does not help because the model's state depends on the whole prefix. Every practical decoder is an approximation.

### 2.2 Greedy as a beam of one

Greedy decoding takes $y_t = \arg\max_v p(v \mid y_{1:t-1})$ at every step (ties to the lowest id) until `eos` or $L$ tokens. It is a beam of size 1. Its failure is commitment: a token that is slightly more likely now can lead to a much less likely continuation. The worked example in section 3 is exactly that case.

### 2.3 The algorithm

Beam search keeps up to $K$ prefixes. One step, in the contract's op order:

1. Call `step_fn(state, y_prev)`; it returns logits `[k, V]` for the $k$ live hypotheses and the new state.
2. Normalize each row with `log_softmax` (`M09.2`): logits are scores, not log-probabilities.
3. Every extension gets $\ell(y_{1:t}) + \log p(v \mid y_{1:t})$: a `[k, V]` table of candidates.
4. Rank **all** $k V$ candidates together, highest first, ties to the lower flat index $b V + v$ (lower beam, then lower token id), and keep the first $K - (\text{finished so far})$ finite ones.
5. A kept candidate ending in `eos` is finished and leaves the beam, keeping its slot: the beam narrows by one. The rest are the next live hypotheses.
6. The state rows of the new hypotheses are their parents' rows: `select_state(new_state, parents)`.

The search stops when no hypothesis is live, or after $L$ tokens, when the live ones are returned unfinished. The result is the finished hypotheses ranked by score.

`select_state` is what makes the search generic. A state is any nesting of tuples, NamedTuples, lists, and dicts whose array leaves (anything with a `shape` of at least one dimension, numpy arrays and Tensors alike) have the hypothesis on axis 0. It gathers `leaf[parents]` for every leaf, rows may repeat (two children of one parent), and everything else is passed through. A Python list is a container, not an array: its elements are searched for arrays, and its own order is not gathered.

Two equivalences pin the algorithm down. With $K = 1$ it is greedy. With $K \ge V^L$ nothing is ever pruned: at step $t$ there are at most $V^{t}$ candidates, all kept, so the result is every possible output in score order, which a test enumerates by brute force.

### 2.4 Length normalization

Every token multiplies the probability by a number below 1, so $\ell$ always prefers shorter outputs. Wu et al. (GNMT) rank finished hypotheses by $\ell(y) / \lvert y \rvert^\alpha$; here $\lvert y \rvert$ counts `eos`. $\alpha = 0$ is the raw log-probability and $\alpha = 1$ the mean per token. During the search all live hypotheses have the same length, so ranking them by $\ell$ or by $s$ is the same thing; the penalty only matters when finished hypotheses of different lengths are compared.

## 3. Worked example by hand

Three tokens, `eos` = 0, `a` = 1, `b` = 2, and a model that is a table of next-token probabilities:

| prefix | $p(\text{eos})$ | $p(a)$ | $p(b)$ |
|---|---|---|---|
| (empty) | 0.1 | 0.5 | 0.4 |
| `a` | 0.2 | 0.1 | 0.7 |
| `b` | 0.9 | 0.05 | 0.05 |

**Greedy** ($L = 2$): `a` (0.5), then after `a` the best is `b` (0.7). Output `a b`, unfinished, $P = 0.35$, $\ell = \ln 0.35 = -1.0498$.

**Beam** ($K = 2$, $L = 2$, $\alpha = 0$). Step 1 ranks `a` 0.5, `b` 0.4, `eos` 0.1 and keeps `a` and `b`. Step 2 ranks all six extensions together:

| candidate | probability |
|---|---|
| `b eos` | $0.4 \times 0.9 = 0.36$ |
| `a b` | $0.5 \times 0.7 = 0.35$ |
| `a eos` | $0.5 \times 0.2 = 0.10$ |
| `a a` | 0.05 |
| `b a`, `b b` | 0.02 each |

It keeps `b eos` (finished) and `a b`, which is cut at $L = 2$ and returned unfinished. Result: `b eos` with $\ell = \ln 0.36 = -1.0217$, then `a b` with $\ln 0.35$. The model was called twice, with 1 and then 2 rows. With $\alpha = 1$ the scores are $\ln 0.36 / 2 = -0.5108$ and $\ln 0.35 / 2 = -0.5249$, the same order. A correct solution reproduces all three results: the beam, greedy, and the penalized scores.

## 4. The interface

```python
@dataclass
class Hypothesis:
    tokens: list[int]; logprob: float; score: float; finished: bool
def select_state(state: Any, idx: ArrayLike) -> Any: ...
def beam_search(step_fn, init_state, bos: int, eos: int, beam_size: int, max_len: int,
                length_penalty: float = 1.0) -> list[Hypothesis]: ...
def greedy_decode(step_fn, init_state, bos: int, eos: int, max_len: int) -> Hypothesis: ...
# step_fn(state, y_prev int64 [k]) -> (logits [k, V], new_state)
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example beam beats greedy | unit | section 3: `b eos` (0.36) then `a b` (0.35), calls `[1, 2]` | pins down the algorithm |
| hand example: greedy commits to `a` | unit | greedy gives `a b`, score equal to logprob | the baseline beam search must beat |
| hand example length penalty | unit | scores $\ln 0.36 / 2$ and $\ln 0.35 / 2$ | the penalty counts `eos` |
| beam one is greedy | differential | a beam of 1 equals the test's own greedy loop on 6 prefix-dependent models | greedy is a special case, not a second code path |
| exhaustive beam equals brute force | differential | all 31 outputs of a $V = 3$, $L = 4$ model, by score, $\alpha \in \{0, 1\}$ | the search is exact when nothing is pruned |
| small beam at $L = 1$ is the top of the exhaustive ranking | property | with $L = 1$ the beam is the $K$ most likely first tokens | ranking over all candidates |
| beam never holds more than beam size | property | the step function never sees more than $K$ rows | the cost bound of beam search |
| `eos` finishes and shrinks the beam | property | `eos` once and last; logprob equals the path sum; the beam never grows back | finished outputs are final |
| search stops when the last hypothesis finishes | boundary | exactly two calls when `eos` is certain at step 2 | no call with zero rows |
| `max_len` returns unfinished | boundary | a model that never says `eos` gives $K$ unfinished outputs of length $L$ | decoding always terminates |
| length penalty changes the winner | unit | $\alpha = 0$ picks `a eos`, $\alpha = 1$ picks `b b b eos`; results ranked by score | why translation systems use a penalty |
| ties go to the lowest id | boundary | uniform logits keep tokens 0 and 1 first | the same order in Python and Rust |
| logits need not be normalized | property | adding a constant per row changes nothing | models return logits |
| masked tokens are never chosen | boundary | `-inf` is never kept, even if the beam is not full | grammar and stop-list masks |
| `logprob` is the sum along the path | property | `logprob` and `score` recomputed from the model | rescoring and comparison across searches |
| `select_state` keeps the structure | unit | NamedTuple, dict, list, repeated rows, non-array leaves; a set raises | any model's state works |
| state follows parents in a real search | property | every state row extends a prefix the search kept | the step function scores the right prefixes |
| validation | boundary | bad sizes, wrong logits shape, nothing finite | caller bugs fail loudly |
| beam decodes a `Seq2Seq` | differential | over `L4.1`'s `Seq2Seq` with Luong attention: beam 1 equals `greedy`, every hypothesis's logprob equals teacher-forced rescoring | the zoo's call site |

### The oracle tests

The model suite [`test_beam.py`](../../../../course/solutions/learner-tests/L4.4/python/tests/l4-4-beam/test_beam.py) uses **oracles**: expected values from an independent computation, not from the implementation. For beam search the oracle is brute force: a tiny model (a table, or logits from a hash of the prefix), every output enumerated for $V = 3$ and $L = 4$, each scored with an independent log-softmax, and compared with an exhaustive beam. It adds greedy equivalence, the section 3 numbers, a model that never says `eos`, a mask, ties, and `select_state` on a NamedTuple inside a dict. Its toy state is an **array** (an object array of prefixes works), because a Python list is searched for arrays, not gathered.

## 5. The solution, step by step

### From first principles

1. **Exact search is out, so bound the work.** $V^L$ sequences cannot be enumerated (section 2.1). Keeping at most $K$ prefixes per step bounds each step to one model call on $\le K$ rows and $K V$ candidates.
2. **Prefixes must be comparable across parents.** A good continuation of a mediocre prefix can beat every continuation of the best prefix (section 3: `b eos` at 0.36 beat `a b` at 0.35 although `a` led). So the $k V$ candidates are ranked *together*, not $K$ per parent, which would also grow the beam to $K^2$.
3. **Scores must be log-probabilities.** Summing over the path requires each step's term to be $\log p$; the model returns logits, so each row is normalized with `log_softmax` first. Adding raw logits would let a row's arbitrary offset decide the ranking.
4. **`eos` ends a hypothesis and spends a slot.** A finished hypothesis can never be extended, and the result should hold $K$ outputs, so each finished one reduces how many live ones are kept: $K - |\text{finished}|$. When that reaches 0, or nothing is live, the search stops before calling the model again.
5. **The search must terminate and return something.** At $L$ tokens the survivors are returned as unfinished hypotheses rather than dropped.
6. **Determinism needs a tie rule.** Ranking ties go to the lower flat index $bV + v$, which a stable sort of the negated scores gives for free. Python and Rust implementations then agree.
7. **The model is a black box with a state.** The only coupling is `step_fn(state, y_prev) -> (logits, state)` with hypotheses on axis 0 of every array in the state. Reordering the state by parent index (`select_state`) keeps the model's view in sync with the search's.
8. **Length normalization is for comparing finished outputs.** Live hypotheses always have the same length, so dividing by $|y|^\alpha$ would not change their order; it is applied only to finished ones, counting `eos`.

### The code, annotated

`select_state` walks containers and gathers array leaves:

```python
def select_state(state, idx):
    idx = np.asarray(idx, dtype=np.int64)
    if isinstance(state, dict):
        return {k: select_state(v, idx) for k, v in state.items()}
    if isinstance(state, tuple):
        parts = [select_state(v, idx) for v in state]
        # A NamedTuple is rebuilt field by field; a plain tuple from the list.
        return type(state)(*parts) if hasattr(state, "_fields") else tuple(parts)
    if isinstance(state, list):
        return [select_state(v, idx) for v in state]   # a container: recurse, do not reorder
    if isinstance(state, (set, frozenset)):
        raise TypeError("select_state: a set has no row order")
    shape = getattr(state, "shape", None)
    if shape is None or len(shape) == 0:
        return state                                    # scalars, None, ints: passed through
    return state[idx]                                   # rows may repeat: two children of one parent
```

Because `L4.1`'s `DecoderState` is a NamedTuple of Tensors (and `None` for unused fields), this one function reorders `h`, `c`, `feed`, `keys`, `mask` and `proj` by parent, and `state.h` still works afterwards.

The search loop, one iteration per generated token:

```python
def beam_search(step_fn, init_state, bos, eos, beam_size, max_len, length_penalty=1.0):
    # ... (beam_size >= 1, max_len >= 1, length_penalty >= 0)
    live_tokens = [[]]                        # one empty prefix
    live_lp = np.zeros(1, dtype=np.float64)
    state = init_state
    y_prev = np.array([bos], dtype=np.int64)
    finished = []
    for t in range(max_len):
        logits, state = step_fn(state, y_prev)
        z = np.asarray(getattr(logits, "data", logits), dtype=np.float64)   # Tensor or ndarray
        k = len(live_tokens)
        # ... (z must be [k, V])
        V = z.shape[1]
        # Normalize each row first: logits are scores, not log-probabilities.
        cand = (live_lp[:, None] + log_softmax(z, axis=-1)).ravel()         # [k * V]
        # Highest first; a stable sort of the negated scores sends ties to the
        # lower flat index b * V + v.
        order = np.argsort(-cand, kind="stable")
        # Every finished hypothesis takes one slot for good: the beam narrows.
        n_keep = min(beam_size - len(finished), int(np.isfinite(cand).sum()))
        new_tokens, new_lp, parents = [], [], []
        for f in order[:n_keep]:
            b, v = divmod(int(f), V)          # parent row, token id
            toks = live_tokens[b] + [v]
            lp = float(cand[f])
            if v == eos:
                finished.append(Hypothesis(toks, lp, _score(lp, len(toks), length_penalty), True))
            else:
                new_tokens.append(toks); new_lp.append(lp); parents.append(b)
        if not new_tokens or len(finished) >= beam_size:
            break                             # never call the model with zero rows
        if t == max_len - 1:
            # Out of length: the survivors are returned as they stand.
            for toks, lp in zip(new_tokens, new_lp):
                finished.append(Hypothesis(toks, lp, _score(lp, len(toks), length_penalty), False))
            break
        state = select_state(state, parents)  # the model's rows follow the kept prefixes
        live_tokens = new_tokens
        live_lp = np.array(new_lp, dtype=np.float64)
        y_prev = np.array([toks[-1] for toks in new_tokens], dtype=np.int64)
    # sorted() is stable: equal scores keep the order they were finalized in.
    return sorted(finished, key=lambda h: -h.score)[:beam_size]
```

Trace it on section 3. Step 1: one row, candidates $\ln(0.1, 0.5, 0.4)$; `n_keep = 2`; kept `a` and `b`, parents `[0, 0]`. Step 2: two rows, six candidates; the top two are `b eos` (finished, slot spent) and `a b`; `t == max_len - 1`, so `a b` is appended unfinished. Two model calls, with 1 and 2 rows.

`_score` is `logprob / n ** length_penalty`, and `greedy_decode` is literally a beam of 1 with `length_penalty=0.0`, so its score equals its log-probability:

```python
def greedy_decode(step_fn, init_state, bos, eos, max_len):
    hyps = beam_search(step_fn, init_state, bos, eos, 1, max_len, length_penalty=0.0)
    if not hyps:
        raise ValueError("greedy_decode: every token is -inf at the first step")
    return hyps[0]
```

Full solution: [`beam.py`](../../../../course/solutions/python/tinyllm/infer/beam.py).

### Why this solution

- **Cost.** At most $L$ model calls on $\le K$ rows each; per step $O(K V)$ for the normalization and $O(K V \log(K V))$ for the full sort, plus the state gather. A partial selection (`argpartition`) would be $O(K V)$, but would need extra work to keep the tie rule; at these sizes the sort is not the bottleneck, the model is.
- **Why rank all $kV$ together rather than top-$K$ per parent?** Per-parent selection keeps $K^2$ rows (pitfall 1), and it can still miss the global best because it does not compare across parents.
- **Why let finished hypotheses spend slots?** The alternative (keep $K$ live rows until all $K$ finished outputs are found) does more model calls and returns a different set; the narrowing beam is the classic formulation and bounds the work by $K L$ rows total.
- **Why a step-function interface?** The search never needs to know the model's architecture, so one implementation serves the toy table, `Seq2Seq` and a transformer with a KV cache. The price is that the state must have hypotheses on axis 0 of every array.
- **Known limitation.** Beam search is not exact; larger beams can even find worse translations (Stahlberg and Byrne). The length penalty partly compensates for the bias toward short outputs.

### Explain it back

1. In section 3, greedy chose `a` first. Why does the beam end up preferring `b eos`?
   <details><summary>Answer</summary>The beam keeps `b` alive at step 1 (0.4, second best) and then compares all six extensions together. $0.4 \times 0.9 = 0.36$ beats every continuation of `a`, the best being $0.5 \times 0.7 = 0.35$.</details>
2. Why is `log_softmax` applied before adding to `live_lp` even though the model "already" ranks tokens?
   <details><summary>Answer</summary>Within a row, logits and log-probabilities rank tokens the same way, but across rows they do not: each row's logits can have a different offset. Summing along a path is only meaningful for log-probabilities.</details>
3. What does `n_keep = min(beam_size - len(finished), isfinite(cand).sum())` protect against, in each of its two terms?
   <details><summary>Answer</summary>The first keeps the total of finished plus live at most $K$, so the beam narrows as hypotheses finish. The second stops `-inf` (masked) candidates from being kept just to fill the beam.</details>
4. Why does the length penalty not affect which live hypotheses are kept?
   <details><summary>Answer</summary>All live hypotheses at step $t$ have exactly $t$ tokens, so dividing every score by $t^\alpha$ preserves their order. It only matters when comparing finished hypotheses of different lengths.</details>
5. What would go wrong if `select_state` were skipped?
   <details><summary>Answer</summary>Row $i$ of the state would still describe the old row $i$'s prefix, while the search records `live_tokens[i]` as a different prefix. The model would score one sentence and the search would credit another.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. keeping the best $K$ tokens of **each** hypothesis | the beam grows to $K^2$ rows, then more | one `argsort` over the flattened `[k * V]` table (step 4) |
| 2. adding raw logits instead of log-probabilities | rows with a larger logit scale win; disagrees with brute force | `log_softmax(z, axis=-1)` before adding to `live_lp` |
| 3. an `eos` hypothesis that keeps going | outputs with tokens after `eos`, a search that never ends early | an `eos` extension goes to `finished`, not to `new_tokens`, and keeps its slot |
| 4. dropping the live hypotheses at `max_len` | an empty result for a model that rarely says `eos` | at `t == max_len - 1` survivors are appended with `finished=False` |
| 5. keeping `-inf` candidates to fill the beam | masked tokens appear in the output | `n_keep` is capped by `np.isfinite(cand).sum()` |
| 6. state rows not gathered by parent | the model scores one prefix and the search records another | `select_state(state, parents)` after every step |
| `eos` left out of the length | scores disagree with the formula | `_score(lp, len(toks), ...)` where `toks` includes `eos` |
| results ranked by logprob, not score | the length penalty has no effect | the final `sorted(..., key=lambda h: -h.score)` |
| ties to the highest index | Python and Rust disagree on near-uniform rows | `np.argsort(-cand, kind="stable")` |
| survivors finalized one step early | outputs one token short | survivors are finalized only at `t == max_len - 1`, after the last token is appended |
| a NamedTuple rebuilt as a tuple | the decoder's `state.h` access fails | `type(state)(*parts)` when the tuple has `_fields` |
| lists not searched | an LSTM's per-layer state list is never reordered | lists are recursed into element by element |
| greedy scored with length penalty 1 | `greedy_decode`'s score is not its logprob | `greedy_decode` passes `length_penalty=0.0` |
| the model called after every hypothesis finished | a call with zero rows, wasted work | `if not new_tokens ...: break` before the next call |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M09.2` | `log_softmax` normalizes each step's logits |
| Back | `L4.1` | `Seq2Seq.decode_step` is a real step function, its `DecoderState` the state |
| Back | `L4.3` | the Luong decoder carries `feed`, one more Tensor for `select_state` to gather |
| Forward | `L6.7` | the zoo decodes seq2seq and transformer checkpoints with beam search and reports beam against greedy |
| Forward | `L5.5` | the encoder-decoder transformer translates with `beam_search` over its own step function |
| Forward | `L8.1` | the sampler is the other way to pick tokens; it shares the tie rule |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `beam_search` | Hugging Face `generate(num_beams=...)` | batched beams across many inputs, `early_stopping` modes, `num_return_sequences` | `transformers/generation/utils.py` (`_beam_search`), `beam_search.py` (`BeamHypotheses`) |
| `select_state` | the KV-cache reorder | gathers the attention cache by beam index every step; with paged attention only block tables are copied | `_reorder_cache` in HF models; vLLM's beam search over forked sequences |
| `length_penalty` | GNMT's $((5 + \lvert y \rvert)/6)^\alpha$ and coverage penalty | a smoother penalty and a term for source words never attended | Wu et al. 2016, section 7; fairseq `sequence_generator.py` |
| exact search for tiny $V^L$ | exact decoding studies | shows the most likely output is often empty: the beam's errors help | Stahlberg and Byrne, "On NMT Search Errors and Model Errors" (2019) |
