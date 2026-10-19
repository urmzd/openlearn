# Sampling and logit processors

## Overview

| | |
|---|---|
| **Module** | `L8.1` · build · Python · Pass 6 · 2 h reading |
| **Solution** | [`sample.py`](../../../../course/solutions/python/tinyllm/infer/sample.py): `SamplingParams` (with `validate`), `request_rng`, `apply_penalties`, `token_logprobs`, `process_logits`, `sampling_distribution`, `sample`, `sampled_entropy`, and the helpers `_seq_sum`, `_softmax_over`, `_kept`; model oracle tests in [`test_sample_oracle.py`](../../../../course/solutions/learner-tests/L8.1/python/tests/l8-1-sample/test_sample_oracle.py) |
| **Contract** | [`course/contracts/py/tinyllm/infer/sample.pyi`](../../../../course/contracts/py/tinyllm/infer/sample.pyi) · the op order, to the bit: [`spec/sampling.md`](../../../../course/contracts/spec/sampling.md) · the generator: [`spec/pcg32.md`](../../../../course/contracts/spec/pcg32.md) |
| **Needs** | `M07.1` (`sample_categorical` is step 11) · `M06.3` (`PCG32` and its `sample` sub-stream) · `M11.1` (`entropy` of what was sampled) · reading: `M09.2` the stable softmax, `M07.6` |
| **Used by** | `L8.2` samples every generated token · `ds.04` checks its C top-k against this top-k · `L8.7` samples from its masked logits · `L8.6` the target distribution of speculative decoding · later: `L10.1` the Rust sampler (same ids on the same logits and seed), `L12.3` GRPO rollouts |
| **Optional depth** | Holtzman et al., "The Curious Case of Neural Text Degeneration" (2020, top-p); Keskar et al., "CTRL" (2019, repetition penalty); Nguyen et al., "Turning Up the Heat: Min-p Sampling" (2024); the OpenAI API reference for `presence_penalty` and `frequency_penalty` |

## Key Takeaways

- Sampling is a fixed pipeline over float64: penalties, then greedy or temperature, then top-k, top-p, min-p (each on what the previous one kept), then softmax, one uniform, and the inverse CDF; the spec fixes the order, the arithmetic, and the draw count, so Python and Rust emit the same token.
- Every sum is a left-to-right loop in ascending id order: Python's `sum()` (compensated since 3.12) and numpy's pairwise `np.sum` give different last bits, and a last bit is enough to flip a token.
- Top-p keeps the token that crosses $p$, the comparison is $\ge$, and ties anywhere go to the lowest id.
- The logprob is taken after the penalties and before temperature and filtering, so it does not depend on how adventurous the sampler was.
- One uniform per sampled token, none per greedy token: the generator's position is a pure function of how many tokens were sampled.

## How to read this chapter

Sections 1 to 3 derive the sampling pipeline of `spec/sampling.md` step by step and run the spec's example by hand. Section 4 fixes the interface and the cases a correct solution must handle. Section 5 walks through the solution and explains why every sum, comparison, and tie rule is written the way it is. Try to predict each step of the solution before reading it.

---

## 1. Why now

The engine has served greedy text since Pass 1: the tracer picks the largest logit. A small trained model, decoded greedily, loops ("the cat sat on the mat. the cat sat on the mat."), and every serving API the system exposes (`L10.5`) accepts `temperature`, `top_p`, `presence_penalty`, and a `seed`. Two engines will run them: this Python sampler, which tests and experiments use, and the Rust engine's (`L10.1`), which serves traffic. If they disagree on a single detail (the order of two filters, how a tie is broken, how a sum is accumulated), the same request with the same seed returns different text from the two, and every parity test, every reproduction of a bad output, and every A/B comparison of the engine against Python becomes noise. This module implements `spec/sampling.md` exactly and proves it on an independent transcription of the spec, the same fixture the Rust port must pass.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $V$ | vocabulary size | integer |
| $x \in \mathbb{R}^V$ | the model's next-token logits | `float32[V]` |
| $l \in \mathbb{R}^V$ | the logits in float64 after each step | `float64[V]` |
| $\mathrm{prompt}$, $\mathrm{out}$ | the request's prompt ids and the ids generated so far | lists of ids |
| $c_i$ | how many times id $i$ occurs in $\mathrm{out}$ | integer |
| $r$, $a_p$, $a_f$ | repetition, presence, and frequency penalties | floats |
| $T$ | temperature | float $\ge 0$ |
| $k$, $p$, $m$ | top-k, top-p, min-p | integer, floats |
| $K$ | the kept set of ids after the filters | set of ids |
| $q_i = e^{l_i - M} / Z$ | step 9: softmax over $K$, $M = \max_K l$, $Z = \sum_K e^{l_i - M}$ | `float64[V]`, 0 outside $K$ |
| $u$ | one uniform on $[0, 1)$ from the request's generator | float |

### 2.1 The pipeline

Steps 1 to 11 of `spec/sampling.md`, in this order and no other:

1. **Widen**: $l_i = \mathrm{f64}(x_i)$.
2. **Repetition** (Hugging Face): for each distinct $i$ in prompt and out, $l_i \leftarrow l_i / r$ if $l_i > 0$, else $l_i \cdot r$.
3. **Presence and frequency** (OpenAI): for each $i$ with $c_i > 0$, $l_i \leftarrow l_i - a_f c_i - a_p$. The **logprob** is $\mathrm{logsoftmax}(l)_i = l_i - M - \log Z$ over all $V$ ids, here.
4. **Greedy**: if $T = 0$, return $\arg\max_i l_i$, ties to the lowest id, with no draw.
5. **Temperature**: $l_i \leftarrow l_i / T$.
6. **Top-k**: order ids by $(l$ descending, id ascending$)$ and keep the first $k$.
7. **Top-p**: compute $q$ over the kept ids; walk them in that order adding $q$ to $s$; keep every id up to and including the first one with $s \ge p$.
8. **Min-p**: compute $q$ over the kept ids; keep those with $q_i \ge m \max_K q$.
9. **Softmax** over $K$, ascending ids.
10. **Draw** exactly one $u$.
11. **Inverse CDF**: walk $K$ in ascending id order adding $q_i$ to $c$; return the first id with $u < c$ (`M07.1`'s `sample_categorical`).

### 2.2 Penalties

The repetition penalty discourages any token already seen. Dividing a positive logit by $r > 1$ makes it smaller, but dividing a negative logit by $r$ moves it toward 0, making the token more likely; that is why the rule multiplies negative logits. It applies once per distinct id, so a token seen ten times is penalized once. Presence and frequency are additive and count only generated tokens: presence is a flat cost for having appeared, frequency grows with the count. Repetition first, then the additive ones: on $l = 3$, $r = 2$, $a_p = 1$ that gives $3/2 - 1 = 0.5$, while the other order gives $(3 - 1)/2 = 1$.

### 2.3 Filters

Each filter sees the distribution the previous one left. Top-k is a hard cap. Top-p (nucleus) keeps the smallest set of most likely tokens whose mass reaches $p$, recomputed on what top-k kept: with $q = [0.4, 0.3, 0.2, 0.1]$ and $k = 2$, id 0 is worth $0.4/0.7 = 0.57$ among the survivors, which alone reaches $p = 0.55$. Min-p keeps tokens at least $m$ times as likely as the best one, so it adapts: a confident distribution keeps few tokens, a flat one keeps many. Masked logits ($-\infty$, from `L8.7`) never enter $K$. Removed ids are $-\infty$ in `process_logits`, not a large negative number: $-10^9$ divided by a later temperature, or exponentiated in another language's float type, can become a small nonzero probability.

### 2.4 Greedy and temperature

$T \to 0$ concentrates all mass on the argmax: with a clear margin, $T = 10^{-3}$ and top-k $= 1$ give the greedy token for any $u$. $T = 0$ itself is defined as greedy, with no division and no draw. Ties go to the lowest id everywhere: in step 4, in the top-k order, and in the inverse CDF's walk.

### 2.5 Exactness and the draw

The inverse CDF maps $u \in [F_{i-1}, F_i)$ to id $i$, an interval of length $q_i$, so $P(i) = q_i$ exactly; `sampling_distribution` returns that $q$, and the tests check the boundaries $u = F_i^-$ and $u = F_i$ for every configuration. The normalizer $Z$ must be a plain running sum. Since Python 3.12, `sum()` of floats uses Neumaier's compensated summation, and `np.sum` adds pairwise: both are more accurate, and both differ from the Rust loop in the last bit of $q$. When $u$ falls within an ulp of a boundary, that flips the token. Accuracy is not the goal here; agreement is. If rounding leaves the final $c$ below $u$, the token is the largest kept id (which always has $q > 0$).

### 2.6 The request generator

A request with seed $s$ draws from $\mathrm{stream}(s, \texttt{sample})$ of `spec/pcg32.md`: `PCG32(s).substream("sample")`, a PCG32 seeded with $\mathrm{splitmix64}(s + 4 \cdot \mathrm{GOLDEN})$ on sequence 4. Exactly one $u$ per sampled token means a decode worker that takes over a request after $n$ sampled tokens skips exactly $n$ uniforms (disaggregated serving, `L10.6`), and speculative decoding's acceptance draws (`M07.6`, `L8.6`) interleave at known positions.

## 3. Worked example by hand

The spec's example: $V = 5$, $x = [1, 3, 2, 3, -1]$, no history, $T = 1$, top-k $= 3$, top-p $= 0.8$, seed 0.

| step | what happens | result |
|---|---|---|
| 1 to 3 | no penalties | $l = [1, 3, 2, 3, -1]$ |
| logprob | $M = 3$, $Z = e^{-2} + 1 + e^{-1} + 1 + e^{-4} = 2.52153$ | ids 1 and 3: $-\ln Z = -0.92487$ |
| 5 | $T = 1$ | unchanged |
| 6 | order $(3.0, 1), (3.0, 3), (2.0, 2), (1.0, 0), (-1.0, 4)$; keep 3 | $\{1, 3, 2\}$ |
| 7 | $q$ over $\{1, 2, 3\}$: $Z = 1 + e^{-1} + 1 = 2.3679$, $q = [0.42232, 0.15536, 0.42232]$; walk 1, 3: $s = 0.42232$, then $0.84464 \ge 0.8$ | $\{1, 3\}$ |
| 9 | softmax over $\{1, 3\}$ | $q_1 = q_3 = 0.5$ |
| 10 | $\mathrm{stream}(0, \texttt{sample})$: child seed `0xF88BB8A8724C81EC`, first uniform | $u = 0.80209$ |
| 11 | $c = 0.5$ at id 1 ($u$ not below), $c = 1.0$ at id 3 ($u < 1$) | **id 3**, logprob $-0.92487$ |

`process_logits` returns $[-\infty, 3, -\infty, 3, -\infty]$ and `sampling_distribution` returns $[0, 0.5, 0, 0.5, 0]$.

## 4. The interface

```python
# python/tinyllm/infer/sample.py
@dataclass
class SamplingParams:
    temperature: float = 1.0; top_k: int = 0; top_p: float = 1.0; min_p: float = 0.0
    repetition_penalty: float = 1.0; presence_penalty: float = 0.0; frequency_penalty: float = 0.0
    seed: Optional[int] = None; max_tokens: int = 128; stop: list[str] = []; logprobs: int = 0
    def validate(self) -> None
def request_rng(seed: int) -> PCG32                                    # stream(seed, "sample")
def apply_penalties(logits, p, history=(), prompt=()) -> NDArray       # steps 1 to 3, float64
def token_logprobs(logits, p, history=(), prompt=()) -> NDArray        # log_softmax after step 3
def process_logits(logits, p, history=(), prompt=()) -> NDArray        # steps 1 to 8, -inf removed
def sampling_distribution(logits, p, history=(), prompt=()) -> NDArray # q, 0 outside K
def sample(logits, p, history, rng, prompt=()) -> tuple[int, float]    # (id, logprob)
def sampled_entropy(logits, p, history=(), prompt=()) -> float         # H(q) in nats (M11.1)
```

`history` is the ids generated so far (the spec's `out`) and `prompt` the prompt ids: the repetition penalty looks at both, presence and frequency at `history` only. Both are separate arguments because the spec needs both.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand example | unit | section 3 end to end: id 3, logprob, processed logits, $q$ | the solution and the reader agree on the spec |
| the hand example's intermediates | unit | top-k's $q$; logprobs ignore temperature and filters | what `L10.5` reports as `logprobs` |
| `request_rng` is the `sample` stream | golden | child seed and first uniform of seed 0 | the same seed gives the same text in Rust |
| repetition penalty, HF semantics | unit | divide positive, multiply negative, once per distinct id | repeated unlikely tokens stay unlikely |
| presence and frequency, OpenAI semantics | unit | counts of generated ids only | OpenAI compatibility |
| penalty order | unit | repetition before presence | the spec's order |
| top-p keeps the crossing token | boundary | exact sums at $p$: $\ge$, and the crossing token stays | the nucleus is never empty |
| top-k ties go to the lowest id | boundary | ties at the cut | deterministic across languages |
| min-p is relative to the top | unit | the cut is $m \max q$, inclusive | min-p adapts to confidence |
| filters compose in spec order | unit | top-p after top-k, min-p after top-p | the order changes the kept set |
| masked logits are never sampled | property | $-\infty$ ids have $q = 0$ and stay $-\infty$ | constrained decoding (`L8.7`) |
| inverse-CDF boundaries | statistical | $u = F_i^-$ gives $i$, $u = F_i$ the next id, for six configurations | $P(i) = q_i$ exactly |
| chi-square against the distribution | statistical | 20000 seeded draws follow $q$ | the sampler and its reported distribution agree |
| one draw per token, none for greedy | unit | draw accounting | disaggregated serving, speculative decoding |
| greedy and its limits | property | $T = 0$, top-k $= 1$, tiny $T$ all give the argmax, ties lowest | greedy parity under the near-tie rule |
| sums are sequential in ascending ids | unit | $q$ and logprobs bitwise equal a plain loop | bit parity with Rust |
| golden ids and logprobs | golden | 12 cases x 12 tokens from an independent transcription | the file `L10.1` is held to |
| entropy of the sampling distribution | unit | $H(q)$ after filtering; 0 when greedy | what `L8.2` logs per token |
| parameters are validated | boundary | out-of-range parameters and logits raise | caller bugs never return a token |

### A model oracle test suite

[`test_sample_oracle.py`](../../../../course/solutions/learner-tests/L8.1/python/tests/l8-1-sample/test_sample_oracle.py) tests against the contract only. Its oracle is the spec itself: steps 1 to 11 transcribed in plain Python (lists, `math.exp`, explicit loops), fed the same uniforms as the sampler through a scripted source, compared on ids and logprobs over random logits with ties, masks, and every processor. It adds the cases a random search rarely hits: sums exactly at $p$, ties at the top-k cut, greedy with no draw. A second transcription is the right oracle because the risk here is not a crash but a subtly different reading of the spec.

## 5. The solution, step by step

### 5.1 From first principles

1. Two engines must produce the same token from the same logits and seed (section 1). So the goal is **reproducibility**, not accuracy: every operation must be one that a Rust loop reproduces bit for bit. That rules out numpy's pairwise sums and Python's compensated `sum()`, and forces float64 and `math.exp` everywhere.
2. Penalties change the model's opinion and the logprob reports it, so the logprob is taken after step 3 and before any sampler knob (temperature, filters). One function, `apply_penalties`, produces that common starting point for everything else.
3. Greedy is a decision, not a limit: $T = 0$ returns the argmax of the penalized logits with no draw, which keeps the generator's position a pure function of the number of sampled tokens.
4. Filters are set operations on ids, each computed on what the previous one kept. Represent the kept set as a list of ids in (logit descending, id ascending) order for the filters, and sort it ascending for the softmax and the CDF walk, as the spec says.
5. Removed ids must be exactly $-\infty$ so no later operation can resurrect them.
6. Sampling is one uniform and `M07.1`'s inverse CDF over $q$; reusing `sample_categorical` makes "$P(i) = q_i$" the same statement `M07.1` already proved.
7. Every derived view (`process_logits`, `sampling_distribution`, `sampled_entropy`, `sample`) calls the same helpers, so they cannot disagree with each other.

### 5.2 The code, annotated

**Sums and softmax.**

```python
def _seq_sum(xs):
    """Left-to-right float64 sum (numpy's cumsum is a plain running loop)."""
    if len(xs) == 0:
        return 0.0
    # the last running total is exactly ((x0 + x1) + x2) + ..., as in Rust
    return float(np.cumsum(np.asarray(xs, dtype=np.float64))[-1])

def _softmax_over(l, keep):
    """Step 9 over the ids in `keep` (ascending): q_i = exp(l_i - M) / Z."""
    ids = sorted(keep)                                 # ascending id order fixes the sum order
    m = max(float(l[i]) for i in ids)                  # shift by the max (M09.2)
    e = [math.exp(float(l[i]) - m) for i in ids]       # math.exp, not np.exp: one libm call per id
    z = _seq_sum(e)
    return [v / z for v in e]
```

`np.cumsum` is used for the left-to-right property, not for speed: unlike `np.sum`, it cannot reorder additions.

**Penalties.**

```python
def apply_penalties(logits, p, history=(), prompt=()):
    l = np.array(logits, dtype=np.float64)  # step 1: a float64 copy
    # ... reject non-1-D, NaN, +inf, all -inf, and out-of-range ids
    r = p.repetition_penalty
    if r != 1.0:
        for i in sorted(set(hist) | set(pr)):  # step 2: once per distinct id
            # dividing a negative logit would move it toward 0: multiply instead
            l[i] = l[i] / r if l[i] > 0 else l[i] * r
    if p.presence_penalty != 0.0 or p.frequency_penalty != 0.0:
        for i, c in sorted(Counter(hist).items()):  # step 3: generated ids only
            l[i] = l[i] - p.frequency_penalty * c - p.presence_penalty
    return l
```

Iterating in sorted order changes nothing numerically here (each id is touched once) but keeps the code a literal transcription of the spec. Section 2.2's example, $l = 3$, $r = 2$, $a_p = 1$: step 2 gives 1.5, step 3 gives 0.5.

**Logprobs.**

```python
def token_logprobs(logits, p, history=(), prompt=()):
    l = apply_penalties(logits, p, history, prompt)    # after step 3, before temperature
    m = float(np.max(l))
    z = _seq_sum([math.exp(float(v) - m) for v in l])  # over all V ids, ascending
    return l - m - math.log(z)
```

In section 3, $M = 3$ and $Z = 2.52153$, so ids 1 and 3 have logprob $-\ln Z = -0.92487$, whatever top-k and top-p do later.

**The filters.**

```python
def _kept(l, p):
    """Steps 5 to 8 on penalized logits: the kept ids, ascending."""
    t = l / p.temperature  # step 5
    # step 6: order by (logit desc, id asc); masked ids never enter
    order = [i for i in sorted(range(t.size), key=lambda i: (-t[i], i)) if t[i] != -np.inf]
    if 0 < p.top_k < len(order):
        order = order[: p.top_k]
    if p.top_p < 1.0:  # step 7: the nucleus, keeping the token that crosses p
        # q over what top-k kept, not over all V
        q = dict(zip(sorted(order), _softmax_over(t, order)))
        s, cut = 0.0, len(order)
        for n, i in enumerate(order):
            s += q[i]
            if s >= p.top_p:          # >=: a sum exactly at p stops here
                cut = n + 1           # n + 1: the crossing token is kept
                break
        order = order[:cut]
    if p.min_p > 0.0:  # step 8: relative to the most likely kept token
        q = dict(zip(sorted(order), _softmax_over(t, order)))
        qmax = max(q.values())
        order = [i for i in order if q[i] >= p.min_p * qmax]
    return sorted(order)
```

Section 3 through it: order is $(1, 3, 2, 0, 4)$ by the key $(-l, \text{id})$, top-k keeps $(1, 3, 2)$, top-p computes $q = (0.42232, 0.15536, 0.42232)$ over ids $(1, 2, 3)$ and walks $1, 3$: $s = 0.42232$, then $0.84464 \ge 0.8$, so `cut = 2` and the kept set is $\{1, 3\}$.

The `q` dict is built with `sorted(order)` because `_softmax_over` returns probabilities in ascending id order; zipping with `order` directly would pair them with the wrong ids.

**The public views.**

```python
def process_logits(logits, p, history=(), prompt=()):
    p.validate()
    l = apply_penalties(logits, p, history, prompt)
    if p.temperature == 0.0:
        return l                                  # greedy: no temperature, no filters
    out = np.full(l.shape, -np.inf)               # removed ids are exactly -inf
    keep = _kept(l, p)
    out[keep] = l[keep] / p.temperature
    return out

def sampling_distribution(logits, p, history=(), prompt=()):
    # ... validate, apply_penalties
    q = np.zeros(l.size)
    if p.temperature == 0.0:
        q[_greedy(l)] = 1.0                       # greedy is a point mass
        return q
    keep = _kept(l, p)
    q[keep] = _softmax_over(l / p.temperature, keep)
    return q

def sample(logits, p, history, rng, prompt=()):
    lp = token_logprobs(logits, p, history, prompt)
    if p.temperature == 0.0:
        p.validate()
        tok = _greedy(apply_penalties(logits, p, history, prompt))   # np.argmax: first max, lowest id
        return tok, float(lp[tok])                # no draw
    q = sampling_distribution(logits, p, history, prompt)
    tok = sample_categorical(q, rng.uniform())  # step 10 and 11: one draw
    return tok, float(lp[tok])
```

In section 3 the request generator is `request_rng(0) = PCG32(0).substream("sample")`, whose first uniform is $0.80209$. `sample_categorical` walks $q = [0, 0.5, 0, 0.5, 0]$ in ascending ids: $c = 0.5$ at id 1 ($u$ is not below), $c = 1.0$ at id 3, so the token is 3 with logprob $-0.92487$. `sampled_entropy` is `entropy(sampling_distribution(...))`, so it measures exactly the distribution that was sampled.

`SamplingParams.validate` rejects bools posing as numbers, non-finite values, `top_p` outside $(0, 1]$, `min_p` outside $[0, 1]$, a non-positive repetition penalty, and negative seeds, so a bad request fails before it produces a token.

Full solution: [`sample.py`](../../../../course/solutions/python/tinyllm/infer/sample.py).

### 5.3 Why this solution

- **Cost.** One sampled token is $O(V \log V)$ for the sort in step 6 plus $O(V)$ for the logprob normalizer, in pure-Python loops. With $V = 49152$ that is milliseconds per token, which is fine for a reference and for tests; serving uses the Rust port.
- **Python loops over numpy.** Vectorised numpy would be faster but reorders additions (`np.sum` is pairwise) and uses its own `exp`. Bit parity with Rust matters more than speed here, and the loops read as the spec.
- **A full sort over `argpartition`.** `np.argpartition` finds the top $k$ in $O(V)$ but does not define the order among ties; the full sort with key $(-l, \text{id})$ does, which is what the tie rule needs.
- **$-\infty$ over a large negative.** It costs nothing and is the only value that stays impossible through any later division or a different float type.
- **Logprob before temperature.** Reporting the penalized model distribution makes logprobs comparable across sampler settings; reporting the sampled distribution would make perplexity depend on `top_p`.

### 5.4 Explain it back

1. Why is $Z$ computed with a left-to-right loop and not `np.sum`?
   <details><summary>Answer</summary>`np.sum` adds pairwise and Python's `sum()` is compensated; both give a different last bit from the Rust engine's running loop. When $u$ lies within an ulp of a CDF boundary, one bit flips the token, and the two engines then diverge for the rest of the generation.</details>
2. Why does top-p keep the token at which the running sum first reaches $p$?
   <details><summary>Answer</summary>The nucleus is the smallest set whose mass is at least $p$; that set must include the token that pushes the sum over. Dropping it can leave a set with mass below $p$, or an empty set when the top token alone exceeds $p$.</details>
3. Why does a greedy step draw no uniform?
   <details><summary>Answer</summary>So the generator's position depends only on how many tokens were sampled. A disaggregated worker that takes over after $n$ sampled tokens skips exactly $n$ uniforms, and speculative decoding's acceptance draws stay at known positions.</details>
4. Why does the repetition penalty multiply negative logits by $r$ instead of dividing?
   <details><summary>Answer</summary>Dividing a negative logit by $r > 1$ moves it toward 0, making the token more likely, which is the opposite of a penalty. Multiplying makes it more negative.</details>
5. In `_kept`, why is top-p's $q$ computed over what top-k kept rather than over all ids?
   <details><summary>Answer</summary>The spec composes filters: each sees the distribution the previous one left. With $q = [0.4, 0.3, 0.2, 0.1]$ and $k = 2$, id 0 is worth $0.57$ among the survivors and alone reaches $p = 0.55$; over all ids it would need id 1 too.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. temperature before the penalties, or presence before repetition | penalized tokens shift by a temperature-dependent amount | `apply_penalties` runs repetition then presence/frequency, and `_kept` divides by $T$ only afterwards |
| 2. top-p with `s > p`, or dropping the token that crosses $p$ | the nucleus loses a token; with $p$ small it can be empty | `s >= p.top_p` and `cut = n + 1` |
| 3. dividing negative logits by $r$, penalizing per occurrence, or counting prompt ids in presence and frequency | repeated unlikely tokens become likely; prompts bias the output | divide if positive else multiply; loop over `set(hist) \| set(pr)`; `Counter(hist)` only for the additive penalties |
| 4. ties to the highest id | Python and Rust disagree on tied logits | sort key `(-t[i], i)`; greedy uses `np.argmax`, the first maximum |
| 5. a greedy step that draws | every later seeded token shifts | `sample` returns before `rng.uniform()` when $T = 0$ |
| 6. logprobs of the filtered, tempered distribution | logprobs change with `top_p`; perplexity from logprobs is wrong | `token_logprobs` uses `apply_penalties` only (step 2) |
| 7. `sum()` or `np.sum` for $Z$ | rare token flips against Rust, never reproducible | every normalizer goes through `_seq_sum` |
| 8. top-p on the unfiltered distribution, or min-p before top-p | a different kept set than the spec's | `_kept` recomputes $q$ on the current `order` and runs top-k, top-p, min-p in that order |
| 9. `PCG32(seed)` instead of the `sample` sub-stream | a seeded request differs from the Rust engine's | `request_rng` returns `PCG32(seed).substream("sample")` |
| 10. removed ids as $-10^9$ instead of $-\infty$ | masked tokens reappear after a later scaling | `process_logits` starts from `np.full(..., -np.inf)`; `_kept` skips $-\infty$ ids |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M07.1` | `sample_categorical` is step 11; `UniformSource` is the generator type |
| Back | `M06.3` | `PCG32(seed).substream("sample")` is the request's generator |
| Back | `M11.1` | `entropy` of the sampled distribution |
| Back | `M09.2` | the shift by the maximum in step 9 |
| Forward | `L8.2` | `generate` samples every token with `sample` and logs `sampled_entropy` |
| Forward | `ds.04` | the C heap top-k is checked against this sampler's top-k on fixture logits |
| Forward | `L10.1` | the Rust sampler, held to `sampler_golden.json` and to this sampler's ids on shared logits |
| Forward | `L8.6` | `sampling_distribution` is the target $p$ of `M07.6`'s acceptance test |
| Forward | `L8.7` | constrained decoding writes $-\infty$ into the logits before `sample` |
| Forward | `L12.3` | GRPO rollouts sample their completions with this sampler |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `sample` | vLLM `Sampler` | batched penalties and top-k/top-p over a whole batch on the GPU, per-request generators | `vllm/v1/sample/sampler.py` |
| `apply_penalties` | Hugging Face `LogitsProcessor`s | one composable processor per rule (`RepetitionPenaltyLogitsProcessor`, `MinPLogitsWarper`) | `transformers/generation/logits_process.py` |
| `process_logits` | llama.cpp samplers | a configurable chain (typical-p, mirostat, DRY) applied in a user-chosen order | `src/llama-sampling.cpp` |
| `request_rng` | SGLang deterministic sampling | per-request seeds that survive batching and preemption | `sglang/srt/sampling/` |
