# Positional encodings: sinusoidal, learned

[Multi-head attention](03-multi-head-attention.md) gives the same output for "321+654" and "123+456" shuffled into any order of keys: permuting the keys changes nothing. For addition that is fatal, because which digit is the ones digit is entirely a matter of position. The RNNs of the earlier chapters knew order for free, by reading one token at a time. The transformer reads all tokens at once and has to be told.

This section builds the two ways this book tells it. The first is the fixed sinusoidal table of the 2017 transformer, used by the [encoder-decoder Transformer](05-encoder-decoder-transformer.md) at the end of this chapter, whose shift property is the seed of [RoPE](../p07-modern-block/03-rope.md) in the modern block. The second is the learned table of [GPT-2](../p06-objectives/01-gpt-decoder-only.md) and [BERT](../p06-objectives/02-bert-masked-lm.md). You will see why the frequencies form a geometric ladder, prove that moving $k$ positions is a fixed rotation, and learn why the angles must be computed in float64.

The section draws on *Mathematics for Computing*: `rotate_pairs` from [Trig, unit circle, 2D rotations, complex numbers, Euler's formula](../../mathematics-for-computing/00-precalculus/02-trig-rotations-and-eulers-formula.md), `geometric` from [Sequences, geometric series, frequency ladders](../../mathematics-for-computing/00-precalculus/03-sequences-and-frequency-ladders.md), `normal_init` from [Variance propagation and initialization](../../mathematics-for-computing/07-probability-statistics/03-variance-propagation-and-initialization.md), and `PCG32` from [Modular arithmetic, hashing, and PCG32](../../mathematics-for-computing/06-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md). From this book it uses the [`Tensor`](../p00-foundations/01-tensor-and-broadcasting-backward.md), the `embedding` op from [The op library and gradcheck_all](../p00-foundations/02-op-library-and-gradcheck.md), and `Module` from [Module system and basic layers](../p00-foundations/04-module-system-and-layers.md). The full solution is [`python/tinyllm/xfmr/pos.py`](../../../../course/solutions/python/tinyllm/xfmr/pos.py) (`sinusoidal_freqs`, `sinusoidal_pe`, `shift_pe`, `SinusoidalPE`, `LearnedPE`), and its contract is [`course/contracts/py/tinyllm/xfmr/pos.pyi`](../../../../course/contracts/py/tinyllm/xfmr/pos.pyi).

> **Learning objectives**
> - Build the sinusoidal table with interleaved sin and cos columns and the frequency ladder $\omega_i = b^{-2i/d}$.
> - Prove that $\text{PE}(p + k)$ is a fixed block rotation of $\text{PE}(p)$ and implement `shift_pe` with one call.
> - Explain why angles are computed in float64 and rounded to float32 once.
> - Build a learned position table as an embedding whose rows get gradient only when used, and honour `offset` in both modules.

## The notation

| Symbol | Meaning | Type / shape |
|---|---|---|
| $p$ | position, counted from 0 | `int` |
| $d$ | model width, even | `int` |
| $i$ | pair index, $0 \le i < d/2$ | `int` |
| $b$ | base, 10000 in the paper | `float` |
| $\omega_i = b^{-2i/d}$ | angular frequency of pair $i$ (radians per position) | `float64[d/2]` |
| $\text{PE}(p)$ | the encoding of position $p$ | `float32[d]` |
| $R(\theta)$ | the 2D rotation by $\theta$ | `float64[2, 2]` |

## Why add a vector at all

Attention is permutation equivariant, so order must enter through the input. The cheapest place is to add a vector that depends only on the position: $x_p = e(\text{token}_p) + \text{PE}(p)$. Adding (not concatenating) keeps the width $d$ that every later layer already expects.

The vector should stay bounded however long the sequence is, so raw $p$ is out. Points on a unit circle, $(\sin p\omega, \cos p\omega)$, are bounded and distinct for nearby $p$ as long as $\omega$ is small enough not to wrap. But one circle cannot separate both neighbours and distant positions: a fast $\omega$ wraps, a slow one barely moves. The answer is $d/2$ circles at a geometric ladder of speeds.

## The sinusoidal table

$$\text{PE}(p)_{2i} = \sin(p\,\omega_i), \qquad \text{PE}(p)_{2i+1} = \cos(p\,\omega_i), \qquad \omega_i = b^{-2i/d}.$$

The frequencies form a geometric sequence with first term 1 and ratio $b^{-2/d}$ (the `geometric` helper). Pair 0 turns one radian per position and repeats every $2\pi$ positions; the last pair turns about $b^{-1}$ radians per position and repeats every $2\pi b$, about 63 000 positions for $b = 10^4$. Like a clock with hands of geometric speeds, the fast pairs tell neighbours apart and the slow ones tell far positions apart; geometric spacing gives every scale of distance the same number of pairs.

Each pair lies on the unit circle, so $\lVert\text{PE}(p)\rVert = \sqrt{d/2}$ for every $p$: unlike adding $p$ itself, the signal never grows with the sequence. The columns are **interleaved**, sin at even columns and cos at odd ones. That is the layout of the paper and of `rotate_pairs`, which makes the shift law below a single call.

There is a precision trap. The angle $p\,\omega_i$ for $p = 10^4$ is about $10^4$ radians. float32 keeps about 7 significant digits, so the angle is off by roughly $10^4 \times 6 \times 10^{-8} \approx 6 \times 10^{-4}$ rad before the sine is even taken. The table is therefore computed in float64 and rounded to float32 at the end.

## A shift is a rotation

The table's most important property comes from the angle-addition formulas:

$$\begin{pmatrix}\sin((p+k)\omega)\\ \cos((p+k)\omega)\end{pmatrix} = \begin{pmatrix}\cos k\omega & \sin k\omega\\ -\sin k\omega & \cos k\omega\end{pmatrix}\begin{pmatrix}\sin p\omega\\ \cos p\omega\end{pmatrix} = R(-k\omega)\begin{pmatrix}\sin p\omega\\ \cos p\omega\end{pmatrix}.$$

So $\text{PE}(p + k)$ is $\text{PE}(p)$ with pair $i$ turned by $-k\omega_i$: one block-diagonal rotation that depends on $k$ and not on $p$. `shift_pe(pe, k, d)` is `rotate_pairs(pe, -k * omega)`.

Two consequences follow. A linear layer can learn to "look $k$ back" the same way at every position. And $\text{PE}(p) \cdot \text{PE}(p+k) = \sum_i \cos(k\,\omega_i)$ depends only on the distance. RoPE applies exactly this rotation to the queries and keys instead of adding it to the input.

## Learned positions

GPT-2 and BERT take the other route and learn a table $P \in \mathbb{R}^{L \times d}$ (`max_len` rows), adding row $p$ at position $p$. It is an embedding indexed by position, initialized $\mathcal{N}(0, 0.02^2)$, and only the rows actually used get gradient. It can learn any pattern, but it knows nothing about positions past $L$: a model trained on 1024 tokens has no row 1024.

## Offsets

Both modules add positions `offset .. offset + T - 1` to an input of length $T$. During training `offset` is 0. When decoding one token at a time with a [KV cache](../p08-inference/02-kv-cache-and-generate.md), the new token sits at position `offset`, the number of tokens before it, and must get that row, not row 0.

## Working it by hand

With $d = 4$ and $b = 10000$, the ladder is $\omega = (10000^0, 10000^{-1/2}) = (1, 0.01)$, and the first three rows are:

| $p$ | $\sin p$ | $\cos p$ | $\sin 0.01p$ | $\cos 0.01p$ |
|---|---|---|---|---|
| 0 | 0 | 1 | 0 | 1 |
| 1 | 0.841471 | 0.540302 | 0.010000 | 0.999950 |
| 2 | 0.909297 | -0.416147 | 0.019999 | 0.999800 |

Now check the shift on pair 0 with $k = 1$. Applying $R(-1)$ to $(\sin 1, \cos 1) = (0.841471, 0.540302)$ gives $(0.841471 \cos 1 + 0.540302 \sin 1,\; -0.841471 \sin 1 + 0.540302 \cos 1) = (0.454649 + 0.454649,\; -0.708073 + 0.291927) = (0.909297, -0.416147)$, which is the $p = 2$ row.

## Designing the API

```python
def sinusoidal_freqs(d: int, base: float = 10000.0) -> NDArray                   # float64 [d/2]
def sinusoidal_pe(T: int, d: int, base: float = 10000.0, offset: int = 0) -> NDArray   # float32 [T, d]
def shift_pe(pe, k: float, d: int, base: float = 10000.0) -> NDArray
class SinusoidalPE(Module):   # no parameters; forward(x, offset=0) = x + table[offset : offset + T]
class LearnedPE(Module):      # weight [max_len, d]; positions(T, offset); forward(x, offset=0)
```

The fixed table is data, not a parameter: storing it as a plain array keeps it out of the gradient and the state_dict. The learned table is the opposite, a parameter read through `embedding`, so that only the rows used get gradient. Both modules take `offset` so the same module serves training (offset 0) and cached decoding.

## Building it step by step

### The ladder

The ladder is the geometric sequence, with the validation that keeps the table meaningful:

```python
def _check_d(d: int) -> None:
    # Pairs need an even width; an odd last column would have no partner.
    if not isinstance(d, (int, np.integer)) or d < 2 or d % 2:
        raise ValueError(f"d must be a positive even integer, got {d}")


def sinusoidal_freqs(d: int, base: float = 10000.0) -> NDArray:
    _check_d(d)
    # base <= 1 would make the ladder flat or increasing: no slow pairs.
    if not (np.isfinite(base) and base > 1.0):
        raise ValueError(f"base must be finite and > 1, got {base}")
    # omega_i = base^(-2i/d): first term 1, ratio base^(-2/d) (M00.3).
    return geometric(1.0, float(base) ** (-2.0 / d), d // 2)
```

The ratio is $b^{-2/d}$, not $b^{-1/d}$: with $d/2$ terms the last one is $b^{-2(d/2-1)/d} \approx b^{-1}$, so the slowest pair repeats every $2\pi b$ positions.

### The table

The table is an outer product of positions and frequencies, then sin and cos written into interleaved columns:

```python
def sinusoidal_pe(T: int, d: int, base: float = 10000.0, offset: int = 0) -> NDArray:
    if T < 0 or offset < 0:
        raise ValueError(f"T and offset must be non-negative, got {T}, {offset}")
    w = sinusoidal_freqs(d, base)
    # Angles in float64: at p = 10^4 a float32 angle is off by about 5e-4 rad.
    # Positions start at `offset` (0 in training), counted from 0, not 1.
    ang = np.arange(offset, offset + T, dtype=np.float64)[:, None] * w[None, :]
    pe = np.empty((T, d), dtype=np.float64)
    pe[:, 0::2] = np.sin(ang)  # PE(p, 2i)
    pe[:, 1::2] = np.cos(ang)  # PE(p, 2i + 1)
    return pe.astype(np.float32)   # round once, after the trig
```

### The shift

The shift is one call, because the layout matches `rotate_pairs`:

```python
def shift_pe(pe: ArrayLike, k: float, d: int, base: float = 10000.0) -> NDArray:
    x = np.asarray(pe)
    if x.shape[-1] != d:
        raise ValueError(f"pe has width {x.shape[-1]}, expected d = {d}")
    # The pair (sin a, cos a) turned by -k*omega is (sin(a + k omega),
    # cos(a + k omega)): a rotation (M00.2) that depends on k, not on p.
    return rotate_pairs(x, -float(k) * sinusoidal_freqs(d, base))
```

`rotate_pairs(x, θ)` maps a pair $(e, o)$ to $(e\cos\theta - o\sin\theta,\; e\sin\theta + o\cos\theta)$. With $e = \sin a$, $o = \cos a$ and $\theta = -k\omega$ the first entry is $\sin a\cos k\omega + \cos a \sin k\omega = \sin(a + k\omega)$: the minus sign is what moves forward in position. Each pair gets its own angle $k\omega_i$; one shared angle would scramble the slow pairs.

### The fixed module

The fixed module precomputes `max_len` rows once and slices:

```python
class SinusoidalPE(Module):
    def __init__(self, max_len: int, d: int, base: float = 10000.0) -> None:
        super().__init__()
        # ...
        # A constant, not a parameter: no gradient, not in the state_dict.
        self.table = sinusoidal_pe(max_len, d, base)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        T = x.shape[-2]
        # ... width check ...
        if offset < 0 or offset + T > self.max_len:
            # Refuse instead of wrapping: a wrapped slice would silently reuse rows.
            raise ValueError(...)
        # Tensor + ndarray: the gradient flows to x and nowhere else.
        return x + self.table[offset : offset + T].astype(x.dtype)
```

### The learned module

The learned module is an embedding table indexed by position:

```python
class LearnedPE(Module):
    def __init__(self, max_len: int, d: int, std: float = 0.02, rng: Any = None) -> None:
        super().__init__()
        # ...
        r = rng if rng is not None else PCG32(0).substream("init")
        self.max_len, self.d = max_len, d
        # N(0, std^2) from the seeded stream (M07.3): the same init in every language.
        self.weight = Tensor(normal_init((max_len, d), std, r), requires_grad=True)

    def positions(self, T: int, offset: int = 0) -> Tensor:
        # ... range check against max_len ...
        # Rows offset .. offset+T-1 of the table, with their gradient.
        return F.embedding(self.weight, np.arange(offset, offset + T))

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        # ... width check ...
        return x + self.positions(x.shape[-2], offset)
```

`F.embedding` is a gather whose backward scatters into exactly the rows read, so rows past the sequence length stay at their initial values. `positions` is public because GPT-2 and BERT want the position rows on their own. The full solution is [`pos.py`](../../../../course/solutions/python/tinyllm/xfmr/pos.py).

### Why this design

Building the table is $O(\text{max\_len} \cdot d)$ once; each forward is a slice and an add, $O(Td)$. The learned table costs $L \cdot d$ parameters and its gradient touches only $T$ rows.

fairseq and some HF code use a half layout: all sines first, then all cosines. That is equally valid but needs a different rotation helper. Interleaving matches the paper and the rotation helper, so the shift law reuses existing code.

Computing rows per call would allow unbounded positions, but costs trig on every step; a fixed `max_len` matches the learned table's contract and makes "past the table" a clear error. Computing in float32 would be faster but wrong at long positions; the cost of float64 is paid once.

There is one tradeoff accepted here: neither encoding extrapolates past `max_len`. The rotation property is the seed of RoPE, which removes that limit for relative distances.

## What can go wrong

A correct solution must reproduce the worked table (frequencies (1, 0.01) and the three rows) so everyone agrees on layout and ladder, and the worked shift (turning PE(1) by $-\omega$ gives PE(2)) to pin down the rotation's sign. At long positions, the table at positions up to 65535, for two settings, must match a 50-digit mpmath reference, so long contexts keep exact positions. The properties argued above must hold: one fixed $R(k)$ for every $p$, negative $k$ included (the relative-position property RoPE builds on); $\text{PE}(p)\cdot\text{PE}(p+k) = \sum_i \cos k\omega_i$; every pair has norm 1; and `offset` gives the tail of a longer table, which the module honours.

For the modules, the sinusoidal module has an empty state_dict and passes gradient through to $x$; the learned module adds rows `offset..` and only those rows get gradient; and its init is a seeded $\mathcal{N}(0, \text{std}^2)$ with default std 0.02, the same in every language. Odd $d$, a base $\le 1$, and positions past the table must raise, so nothing wraps around silently. The model suite [`test_pos.py`](../../../../course/solutions/learner-tests/L5.4/python/tests/l5-4-pos/test_pos.py) uses the formula itself as the oracle (Python's `math` in float64, including positions $p \ge 10^4$).

The common mistakes and how the code avoids each:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| sin and cos concatenated (or swapped) instead of interleaved | the shift law fails; checkpoints from other code disagree | writes `pe[:, 0::2] = sin`, `pe[:, 1::2] = cos` |
| exponent $i/d$ instead of $2i/d$ | the ladder ends at $b^{-1/2}$: far positions alias | ratio `base ** (-2.0 / d)` over `d // 2` terms |
| angles in float32 | rows past a few thousand off in the 4th decimal | angles built from a float64 `arange`, cast to float32 only at the end |
| the shift turned the wrong way, or by the same angle for every pair | `shift_pe` goes back in position, or scrambles pairs | passes the per-pair vector `-k * omega` to `rotate_pairs` |
| positions counted from 1 | every row one late; HF and torch disagree | `np.arange(offset, offset + T)` starts at 0 when `offset = 0` |
| `offset` ignored | incremental decoding gives every new token position 0 | both modules slice rows `offset .. offset + T - 1` |
| the fixed table registered as a parameter | it trains and lands in the checkpoint | `self.table` is a plain ndarray |
| learned rows read as constants | `wpe` never learns | rows gathered with `F.embedding(self.weight, ...)` |
| `std` ignored by the learned table | every table starts at 0.02 | `normal_init((max_len, d), std, r)` uses the argument |
| an odd width accepted | the last feature has no partner | `_check_d` rejects odd or tiny `d` |

## Where this leads

The [encoder-decoder Transformer](05-encoder-decoder-transformer.md) applies `SinusoidalPE` to both the source and the target embeddings; this is how the chapter's addition model reads digit positions. `LearnedPE` is [GPT-2](../p06-objectives/01-gpt-decoder-only.md)'s `wpe` and [BERT](../p06-objectives/02-bert-masked-lm.md)'s position embedding. [RoPE](../p07-modern-block/03-rope.md) rotates queries and keys by the same ladder instead of adding a vector, and [Whisper decoding](../p14-speech/03-whisper-decoder-and-weights.md) calls this code through its declared contract.

The production equivalents:

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `sinusoidal_pe` | fairseq `SinusoidalPositionalEmbedding` | the half layout (all sin, then all cos) and a padding offset | `fairseq/modules/sinusoidal_positional_embedding.py` |
| `shift_pe` | RoPE | the rotation applied inside attention, so scores depend on $p - q$ only | [RoPE](../p07-modern-block/03-rope.md); Su et al. 2021 |
| `LearnedPE` | HF `GPT2Model.wpe`, `BertEmbeddings.position_embeddings` | absolute positions capped at `n_positions` | `transformers/models/gpt2/modeling_gpt2.py` |
| no position at all | ALiBi, NoPE | a distance penalty on scores, or causality alone, for length extrapolation | Press et al. 2022; Kazemnejad et al. 2023 |

### Further reading

- Vaswani et al., "Attention Is All You Need" (2017), section 3.5.
- Kazemnejad, "Transformer Architecture: The Positional Encoding" (2019).
- Su et al., "RoFormer" (2021), for where the rotation goes next.

## Key terms

- **positional encoding**: a vector depending only on position, added to the token embedding so attention can see order.
- **frequency ladder**: the geometric sequence $\omega_i = b^{-2i/d}$ of angular speeds, from 1 down to about $b^{-1}$.
- **interleaved layout**: sin at column $2i$ and cos at column $2i+1$, the pair layout of the paper.
- **shift law**: $\text{PE}(p+k)$ equals $\text{PE}(p)$ with each pair rotated by $-k\omega_i$, independent of $p$.
- **learned position table**: an embedding of `max_len` rows indexed by position, trained like any parameter.
- **offset**: the absolute position of the first input token, nonzero during cached decoding.

## Summary

- Attention sees its keys as a set, so position must be added to the input: $x_p = e(\text{token}_p) + \text{PE}(p)$.
- Pair $i$ of $\text{PE}(p)$ is the point $(\sin p\,\omega_i, \cos p\,\omega_i)$ with $\omega_i = 10000^{-2i/d}$, a geometric ladder from 1 radian per position down to about $10^{-4}$.
- Moving $k$ positions is a fixed rotation of every pair, the same for every $p$, so $\text{PE}(p) \cdot \text{PE}(p + k)$ depends only on $k$.
- Angles must be computed in float64: at $p = 10^4$ a float32 angle is already off in the fourth decimal.
- A learned table is just an embedding indexed by position: it adds rows `offset ..` and only those rows get gradient.

## Review questions

1. Why must the ratio of the ladder be $b^{-2/d}$ and not $b^{-1/d}$?
   <details><summary>Answer</summary>There are $d/2$ frequencies. With ratio $b^{-2/d}$ the last is about $b^{-1}$, so the slowest pair has period $2\pi b$. With $b^{-1/d}$ the ladder stops at about $b^{-1/2}$, the slowest period is only $2\pi\sqrt b \approx 630$ positions, and far positions alias.</details>
2. Why is the shift a rotation by $-k\omega_i$ rather than $+k\omega_i$?
   <details><summary>Answer</summary>The pair is stored as (sin, cos), not (cos, sin). `rotate_pairs` with angle $\theta$ sends $(\sin a, \cos a)$ to $(\sin(a - \theta), \cos(a - \theta))$, so moving forward by $k\omega$ needs $\theta = -k\omega$.</details>
3. Why is `SinusoidalPE.table` a plain array and not a `Tensor` with `requires_grad`?
   <details><summary>Answer</summary>It is a fixed function of position. As a parameter it would be updated by the optimizer (no longer sinusoidal, so the shift law breaks) and saved in every checkpoint for no reason.</details>
4. What goes wrong in cached decoding if `offset` is ignored?
   <details><summary>Answer</summary>Each new token is fed alone, so $T = 1$, and without `offset` it always gets row 0. Every generated token then looks like the first token of the sequence.</details>
5. Why does `LearnedPE` read its rows through `F.embedding` instead of slicing `weight.data`?
   <details><summary>Answer</summary>Slicing the raw array would give a constant with no path back to `weight`, so the table would never learn. `embedding` records the gather and scatters the gradient into the rows used.</details>
