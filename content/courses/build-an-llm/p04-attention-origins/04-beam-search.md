<!-- ol:module L4.4 -->
# Beam search, generic over a step function

## Overview

| | |
|---|---|
| **Module** | `L4.4` · build · Python · Pass 4 · 3 to 4 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/infer/beam.py`: `Hypothesis`, `select_state`, `beam_search`, `greedy_decode`; and your own oracle tests in `python/tests/l4-4-beam/` |
| **Contract** | [`course/contracts/py/tinyllm/infer/beam.pyi`](../../../../course/contracts/py/tinyllm/infer/beam.pyi) |
| **Tests** | `course/tests/L4.4/test_beam.py` (what they check: section 4); the oracle is exhaustive enumeration of every output of a toy model; your tests are graded by mutation, threshold 0.80 with every pitfall fault required |
| **Needs** | `M09.2` `log_softmax` · `L4.1` `Seq2Seq.decode_step` and `L4.3` `LuongAttention` (the real decoder the tests search over) · reading: `S-M06a` (counting sequences) (or `--ref-deps`) |
| **Used by** | later: `L6.7` the zoo's beam decoding of seq2seq and transformer checkpoints (joins the registry with B7) · later: `L5.5` |
| **Milestone** | `MS-L4` (`{tinyllm} translate --beam 5` must beat greedy) |
| **Optional depth** | Graves, "Sequence Transduction with Recurrent Neural Networks" (2012), section 3.2; Wu et al., "Google's Neural Machine Translation System" (2016), section 7 (length penalty); Meister, Cotterell, and Vieira, "If beam search is the answer, what was the question?" (EMNLP 2020) |

## Key Takeaways

- Greedy decoding commits to the best next token and can miss the best sequence; beam search keeps the `beam_size` best prefixes and finds it in the worked example, 0.36 against greedy's 0.35 (`test_hand_example_beam_beats_greedy`, `test_hand_example_greedy_commits_to_a`).
- All `k * V` extensions are ranked **together**, so the beam never holds more than `beam_size` rows (`test_beam_never_holds_more_than_beam_size`).
- A beam of 1 is greedy decoding, and a beam of at least $V^L$ is exact search; both are checked against independent oracles (`test_beam_one_is_greedy`, `test_exhaustive_beam_equals_brute_force`).
- The model is only a step function: its state is gathered by parent with `select_state`, so the same search decodes a toy table and a real `Seq2Seq` (`test_state_follows_parents_in_a_real_search`, `test_beam_decodes_a_seq2seq`).
- Summed log-probability prefers short outputs; dividing by `len ** length_penalty` can change the winner (`test_length_penalty_changes_the_winner`).

## How to work this chapter

```bash
ol start L4.4              # stubs beam.py; prints your test path and rung (R5)
ol tests L4.4              # the course tests
# write your oracle tests in python/tests/l4-4-beam/ (section 4 lists what to cover), then:
ol check L4.4              # course tests and the mutation grade of your tests
ol mutate L4.4             # the full grade, cached by your test files' hash
ol diff  L4.4              # after passing: your code against the reference
```

---

## 1. Why now

Your seq2seq model (`L4.1`) can translate, but only greedily: at each step it takes the single most likely token and never looks back. On the reversal task that is fine, because the model is nearly certain at every step. On the dates task of `MS-L4`, where the first digit of the year is often uncertain, a greedy decoder commits to a likely-looking first token and then has to live with a worse whole output. This module writes the search that fixes it, once, as a function of a step function, so that later the zoo (`L6.7`) uses the same code for seq2seq and transformer checkpoints. The model and the search are separate pieces with a narrow interface between them, which is how production decoders (vLLM's beam search, Hugging Face's `generate`) are built too.

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

It keeps `b eos` (finished) and `a b`, which is cut at $L = 2$ and returned unfinished. Result: `b eos` with $\ell = \ln 0.36 = -1.0217$, then `a b` with $\ln 0.35$. The model was called twice, with 1 and then 2 rows. With $\alpha = 1$ the scores are $\ln 0.36 / 2 = -0.5108$ and $\ln 0.35 / 2 = -0.5249$, the same order. These are `test_hand_example_beam_beats_greedy`, `test_hand_example_greedy_commits_to_a`, and `test_hand_example_length_penalty`.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_beam_beats_greedy` | unit | section 3: `b eos` (0.36) then `a b` (0.35), calls `[1, 2]` | you and the test agree on the algorithm |
| `test_hand_example_greedy_commits_to_a` | unit | greedy gives `a b`, score equal to logprob | the baseline `MS-L4` compares against |
| `test_hand_example_length_penalty` | unit | scores $\ln 0.36 / 2$ and $\ln 0.35 / 2$ | the penalty counts `eos` |
| `test_beam_one_is_greedy` | differential | a beam of 1 equals the test's own greedy loop on 6 prefix-dependent models | greedy is a special case, not a second code path |
| `test_exhaustive_beam_equals_brute_force` | differential | all 31 outputs of a $V = 3$, $L = 4$ model, by score, $\alpha \in \{0, 1\}$ | the search is exact when nothing is pruned |
| `test_small_beam_is_a_prefix_of_the_exhaustive_ranking_at_step_one` | property | with $L = 1$ the beam is the $K$ most likely first tokens | ranking over all candidates |
| `test_beam_never_holds_more_than_beam_size` | property | the step function never sees more than $K$ rows | the cost bound of beam search |
| `test_eos_finishes_and_shrinks_the_beam` | property | `eos` once and last; logprob equals the path sum; the beam never grows back | finished outputs are final |
| `test_search_stops_when_the_last_hypothesis_finishes` | boundary | exactly two calls when `eos` is certain at step 2 | no call with zero rows |
| `test_max_len_returns_unfinished` | boundary | a model that never says `eos` gives $K$ unfinished outputs of length $L$ | decoding always terminates |
| `test_length_penalty_changes_the_winner` | unit | $\alpha = 0$ picks `a eos`, $\alpha = 1$ picks `b b b eos`; results ranked by score | why translation systems use a penalty |
| `test_ties_go_to_the_lowest_id` | boundary | uniform logits keep tokens 0 and 1 first | the same order in Python and Rust |
| `test_logits_need_not_be_normalized` | property | adding a constant per row changes nothing | models return logits |
| `test_masked_tokens_are_never_chosen` | boundary | `-inf` is never kept, even if the beam is not full | grammar and stop-list masks |
| `test_logprob_is_the_sum_along_the_path` | property | `logprob` and `score` recomputed from the model | rescoring and comparison across searches |
| `test_select_state_keeps_the_structure` | unit | NamedTuple, dict, list, repeated rows, non-array leaves; a set raises | any model's state works |
| `test_state_follows_parents_in_a_real_search` | property | every state row extends a prefix the search kept | the step function scores the right prefixes |
| `test_validation` | boundary | bad sizes, wrong logits shape, nothing finite | caller bugs fail loudly |
| `test_beam_decodes_a_seq2seq` | differential | over `L4.1`'s `Seq2Seq` with Luong attention: beam 1 equals `greedy`, every hypothesis's logprob equals teacher-forced rescoring | the zoo's call site |

### Your graded tests (rung R5)

Rung R5 asks for **oracles**: tests whose expected values come from an independent computation, not from your implementation. For beam search the oracle is brute force: write a tiny model (a table, or logits from a hash of the prefix), enumerate every output for $V = 3$ and $L = 4$, score each with your own log-softmax, and compare with an exhaustive beam. Add greedy equivalence, the section 3 numbers, a model that never says `eos`, a mask, ties, and `select_state` on a NamedTuple inside a dict. Keep the state an **array** (an object array of prefixes works): a Python list is searched for arrays, not gathered. Import only `tinyllm.infer.beam`. `ol check L4.4` requires a mutation score of at least 0.80 with every pitfall fault killed.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. keeping the best $K$ tokens of **each** hypothesis | the beam grows to $K^2$ rows, then more | `test_beam_never_holds_more_than_beam_size` (mutant `s01`) |
| 2. adding raw logits instead of log-probabilities | rows with a larger logit scale win; disagrees with brute force | `test_logits_need_not_be_normalized`, `test_exhaustive_beam_equals_brute_force` (mutant `s02`) |
| 3. an `eos` hypothesis that keeps going | outputs with tokens after `eos`, a search that never ends early | `test_eos_finishes_and_shrinks_the_beam` (mutant `s04`) |
| 4. dropping the live hypotheses at `max_len` | an empty result for a model that rarely says `eos` | `test_max_len_returns_unfinished` (mutant `s05`) |
| 5. keeping `-inf` candidates to fill the beam | masked tokens appear in the output | `test_masked_tokens_are_never_chosen` (mutant `s09`) |
| 6. state rows not gathered by parent | the model scores one prefix and the search records another | `test_state_follows_parents_in_a_real_search`, `test_beam_decodes_a_seq2seq` (mutant `s03`) |
| `eos` left out of the length | scores disagree with the formula | `test_hand_example_length_penalty` (mutant `s06`) |
| results ranked by logprob, not score | the length penalty has no effect | `test_length_penalty_changes_the_winner` (mutant `s07`) |
| ties to the highest index | Python and Rust disagree on near-uniform rows | `test_ties_go_to_the_lowest_id` (mutant `s08`) |
| survivors finalized one step early | outputs one token short | `test_max_len_returns_unfinished` (mutant `s11`) |
| a NamedTuple rebuilt as a tuple | the decoder's `state.h` access fails | `test_select_state_keeps_the_structure` (mutant `s13`) |
| lists not searched | an LSTM's per-layer state list is never reordered | `test_select_state_keeps_the_structure` (mutant `s14`) |
| greedy scored with length penalty 1 | `greedy_decode`'s score is not its logprob | `test_hand_example_greedy_commits_to_a` (mutant `s15`) |
| the model called after every hypothesis finished | a call with zero rows, wasted work | `test_search_stops_when_the_last_hypothesis_finishes` (mutant `s16`) |

## 6. Where it's used next
| Forward | `L5.5` | Registered call site uses this module. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M09.2` | `log_softmax` normalizes each step's logits |
| Back | `L4.1` | `Seq2Seq.decode_step` is the step function of the course test, its `DecoderState` the state |
| Back | `L4.3` | the Luong decoder carries `feed`, one more Tensor for `select_state` to gather |
| Forward | `L6.7` | the zoo decodes seq2seq and transformer checkpoints with beam search and reports beam against greedy |
| Forward | `L8.1` | the sampler is the other way to pick tokens; it shares the tie rule |

If you skip this module, `L6.7` stops with `BLOCKED ... needs L4.4` once it lands: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `beam_search` | Hugging Face `generate(num_beams=...)` | batched beams across many inputs, `early_stopping` modes, `num_return_sequences` | `transformers/generation/utils.py` (`_beam_search`), `beam_search.py` (`BeamHypotheses`) |
| `select_state` | the KV-cache reorder | gathers the attention cache by beam index every step; with paged attention only block tables are copied | `_reorder_cache` in HF models; vLLM's beam search over forked sequences |
| `length_penalty` | GNMT's $((5 + \lvert y \rvert)/6)^\alpha$ and coverage penalty | a smoother penalty and a term for source words never attended | Wu et al. 2016, section 7; fairseq `sequence_generator.py` |
| exact search for tiny $V^L$ | exact decoding studies | shows the most likely output is often empty: the beam's errors help | Stahlberg and Byrne, "On NMT Search Errors and Model Errors" (2019) |
