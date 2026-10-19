# Positional encodings: sinusoidal, learned

## Overview

| | |
|---|---|
| **Module** | `L5.4` · build · Python · Pass 5 · 1 h reading |
| **Solution** | [`python/tinyllm/xfmr/pos.py`](../../../../course/solutions/python/tinyllm/xfmr/pos.py): `sinusoidal_freqs`, `sinusoidal_pe`, `shift_pe`, `SinusoidalPE`, `LearnedPE`; model oracle tests in [`learner-tests/L5.4/.../test_pos.py`](../../../../course/solutions/learner-tests/L5.4/python/tests/l5-4-pos/test_pos.py) |
| **Contract** | [`course/contracts/py/tinyllm/xfmr/pos.pyi`](../../../../course/contracts/py/tinyllm/xfmr/pos.pyi) |
| **Needs** | `M00.2` `rotate_pairs` · `M00.3` `geometric` · `M07.3` `normal_init` · `M06.3` `PCG32` · `L0.1` `Tensor` · `L0.2` `embedding` · `L0.4` `Module` |
| **Used by** | `L5.5` the transformer's sinusoidal positions (this is how the Part 5 addition model reads digit positions) · `L6.1` GPT-2's learned `wpe` · `L6.2` BERT's learned position embeddings · `L14.3` Whisper decoding |
| **Optional depth** | Vaswani et al. (2017), section 3.5; Kazemnejad, "Transformer Architecture: The Positional Encoding" (2019); Su et al., "RoFormer" (2021), for where the rotation goes next |

## Key Takeaways

- Attention sees its keys as a set, so position must be added to the input: $x_p = e(\text{token}_p) + \text{PE}(p)$.
- Pair $i$ of $\text{PE}(p)$ is the point $(\sin p\,\omega_i, \cos p\,\omega_i)$ with $\omega_i = 10000^{-2i/d}$, a geometric ladder from 1 radian per position down to about $10^{-4}$.
- Moving $k$ positions is a fixed rotation of every pair, the same for every $p$, so $\text{PE}(p) \cdot \text{PE}(p + k)$ depends only on $k$.
- Angles must be computed in float64: at $p = 10^4$ a float32 angle is already off in the fourth decimal.
- A learned table is just an embedding indexed by position: it adds rows `offset ..` and only those rows get gradient.

## How to read this chapter

Sections 1 to 3 build both encodings from first principles and work the sinusoidal table and its shift by hand. Section 4 fixes the interface and lists what a correct solution must handle. Section 5 walks through the solution function by function and explains why each choice is the right one. Try to predict each step before reading it.

---

## 1. Why now

Multi-head attention (`L5.3`) gives the same output for "321+654" and "123+456" shuffled into any order of keys: permuting the keys changes nothing. For addition that is fatal, because which digit is the ones digit is all a matter of position. The RNNs of Parts 3 and 4 knew order for free, by reading one token at a time; the transformer reads all tokens at once and has to be told. This module gives the two ways the course uses: the fixed sinusoidal table of the 2017 transformer (`L5.5`), whose shift property is the seed of RoPE in Part 7, and the learned table of GPT-2 (`L6.1`) and BERT (`L6.2`).

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $p$ | position, counted from 0 | `int` |
| $d$ | model width, even | `int` |
| $i$ | pair index, $0 \le i < d/2$ | `int` |
| $b$ | base, 10000 in the paper | `float` |
| $\omega_i = b^{-2i/d}$ | angular frequency of pair $i$ (radians per position) | `float64[d/2]` |
| $\text{PE}(p)$ | the encoding of position $p$ | `float32[d]` |
| $R(\theta)$ | the 2D rotation by $\theta$ (`M00.2`) | `float64[2, 2]` |

### 2.1 The sinusoidal table

$$\text{PE}(p)_{2i} = \sin(p\,\omega_i), \qquad \text{PE}(p)_{2i+1} = \cos(p\,\omega_i), \qquad \omega_i = b^{-2i/d}.$$

The frequencies form a geometric sequence with first term 1 and ratio $b^{-2/d}$ (`M00.3`'s `geometric`). Pair 0 turns one radian per position and repeats every $2\pi$ positions; the last pair turns about $b^{-1}$ radians per position and repeats every $2\pi b$, about 63 000 positions for $b = 10^4$. Like the digits of a clock with hands of geometric speeds, the fast pairs tell neighbours apart and the slow ones tell far positions apart. Each pair lies on the unit circle, so $\lVert\text{PE}(p)\rVert = \sqrt{d/2}$ for every $p$: unlike adding $p$ itself, the signal never grows with the sequence. The columns are **interleaved**: sin at even columns, cos at odd ones, the layout of the paper and of `M00.2`'s pairs.

Precision: the angle $p\,\omega_i$ for $p = 10^4$ is about $10^4$ radians. float32 keeps about 7 significant digits, so the angle is off by roughly $10^4 \times 6 \times 10^{-8} \approx 6 \times 10^{-4}$ rad before the sine is even taken. The table is computed in float64 and rounded to float32 at the end.

### 2.2 A shift is a rotation

By the angle-addition formulas (`M00.2`),

$$\begin{pmatrix}\sin((p+k)\omega)\\ \cos((p+k)\omega)\end{pmatrix} = \begin{pmatrix}\cos k\omega & \sin k\omega\\ -\sin k\omega & \cos k\omega\end{pmatrix}\begin{pmatrix}\sin p\omega\\ \cos p\omega\end{pmatrix} = R(-k\omega)\begin{pmatrix}\sin p\omega\\ \cos p\omega\end{pmatrix}.$$

So $\text{PE}(p + k)$ is $\text{PE}(p)$ with pair $i$ turned by $-k\omega_i$: one block-diagonal rotation that depends on $k$ and not on $p$. `shift_pe(pe, k, d)` is `rotate_pairs(pe, -k * omega)`. Two consequences: a linear layer can learn to "look $k$ back" the same way at every position, and $\text{PE}(p) \cdot \text{PE}(p+k) = \sum_i \cos(k\,\omega_i)$ depends only on the distance. RoPE (`L7.3`) applies exactly this rotation to the queries and keys instead of adding it to the input.

### 2.3 Learned positions

GPT-2 and BERT learn a table $P \in \mathbb{R}^{L \times d}$ (`max_len` rows) and add row $p$ at position $p$. It is an embedding indexed by position (`L0.2`'s `embedding`), initialized $\mathcal{N}(0, 0.02^2)$ (`M07.3`), and only the rows actually used get gradient. It can learn any pattern, but it knows nothing about positions past $L$: a model trained on 1024 tokens has no row 1024.

### 2.4 Offsets

Both modules add positions `offset .. offset + T - 1` to an input of length $T$. During training `offset` is 0. When decoding one token at a time with a cache (`L8.2`), the new token sits at position `offset` = the number of tokens before it, and must get that row, not row 0.

## 3. Worked example by hand

$d = 4$, $b = 10000$: $\omega = (10000^0, 10000^{-1/2}) = (1, 0.01)$.

| $p$ | $\sin p$ | $\cos p$ | $\sin 0.01p$ | $\cos 0.01p$ |
|---|---|---|---|---|
| 0 | 0 | 1 | 0 | 1 |
| 1 | 0.841471 | 0.540302 | 0.010000 | 0.999950 |
| 2 | 0.909297 | -0.416147 | 0.019999 | 0.999800 |

Shift check, pair 0, $k = 1$: $R(-1)$ applied to $(\sin 1, \cos 1) = (0.841471, 0.540302)$ gives $(0.841471 \cos 1 + 0.540302 \sin 1,\; -0.841471 \sin 1 + 0.540302 \cos 1) = (0.454649 + 0.454649,\; -0.708073 + 0.291927) = (0.909297, -0.416147)$, the $p = 2$ row.

## 4. The interface

```python
def sinusoidal_freqs(d: int, base: float = 10000.0) -> NDArray                   # float64 [d/2]
def sinusoidal_pe(T: int, d: int, base: float = 10000.0, offset: int = 0) -> NDArray   # float32 [T, d]
def shift_pe(pe, k: float, d: int, base: float = 10000.0) -> NDArray
class SinusoidalPE(Module):   # no parameters; forward(x, offset=0) = x + table[offset : offset + T]
class LearnedPE(Module):      # weight [max_len, d]; positions(T, offset); forward(x, offset=0)
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked table | unit | section 3: frequencies (1, 0.01) and the three rows | everyone agrees on layout and ladder |
| the hand-worked shift | unit | turning PE(1) by $-\omega$ gives PE(2) | the rotation's sign |
| long positions against high precision | golden | the table at positions up to 65535 for two settings against 50-digit mpmath | long contexts keep exact positions |
| a shift is one fixed rotation | property | one $R(k)$ for every $p$, negative $k$ too | the relative-position property RoPE builds on |
| dot products depend only on the offset | property | $\text{PE}(p)\cdot\text{PE}(p+k) = \sum_i \cos k\omega_i$ | similarity by distance |
| every pair on the unit circle | property | each pair has norm 1 | the signal does not grow with $p$ |
| `offset` continues the table | property | `offset` gives the tail of a longer table; the module honours it | incremental decoding (`L8.2`) |
| the sinusoidal module has no parameters | unit | empty state_dict; gradient passes through to $x$ | nothing to train or checkpoint |
| learned rows are added and get gradient | unit | rows `offset..` added; only those get gradient | GPT-2's and BERT's tables train |
| learned init | statistical | seeded $\mathcal{N}(0, \text{std}^2)$, default 0.02 | the same init in every language |
| bad arguments | boundary | odd $d$, base $\le 1$, positions past the table | no silent wrap-around |

A model test suite that uses the formula itself as the oracle (Python's `math` in float64, including positions $p \ge 10^4$) is in [`test_pos.py`](../../../../course/solutions/learner-tests/L5.4/python/tests/l5-4-pos/test_pos.py).

## 5. The solution, step by step

### 5.1 From first principles

1. Attention is permutation equivariant (`L5.3`), so order must enter through the input. The cheapest place is to add a vector that depends only on the position: $x_p = e(\text{token}_p) + \text{PE}(p)$. Adding (not concatenating) keeps the width $d$ that every later layer already expects.
2. The vector should stay bounded however long the sequence is, so raw $p$ is out. Points on a unit circle, $(\sin p\omega, \cos p\omega)$, are bounded and distinct for nearby $p$ as long as $\omega$ is small enough not to wrap.
3. One circle cannot separate both neighbours and distant positions: a fast $\omega$ wraps, a slow one barely moves. So use $d/2$ circles at a geometric ladder of speeds, $\omega_i = b^{-2i/d}$, from 1 down to $b^{-1}$ (for $d = 4$: $1$ and $0.01$, the worked example). Geometric spacing gives every scale of distance the same number of pairs.
4. The pair is laid out as sin at column $2i$ and cos at column $2i+1$, which is exactly the pair layout `rotate_pairs` (`M00.2`) uses. That choice makes the shift law a single call.
5. Angle addition (section 2.2) turns "move $k$ positions" into "rotate pair $i$ by $-k\omega_i$", independent of $p$. So `shift_pe` needs no position, only $k$ and the ladder.
6. The angle $p\omega_i$ is large for large $p$, and float32 loses about $p \times 6 \times 10^{-8}$ radians. So the table is computed in float64 and rounded once, at the end.
7. The fixed table is data, not a parameter: storing it as a plain array keeps it out of the gradient and the state_dict. The learned table is the opposite: a parameter read through `embedding`, so that only the rows used get gradient.
8. Both modules take `offset` so the same module serves training (offset 0) and cached decoding, where a new token sits at position `offset`.

### 5.2 The code, annotated

The ladder is `M00.3`'s geometric sequence, with the validation that keeps the table meaningful:

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

`F.embedding` is a gather whose backward scatters into exactly the rows read, so rows past the sequence length stay at their initial values. `positions` is public because GPT-2 and BERT (`L6.1`, `L6.2`) want the position rows on their own.

Full solution: [`pos.py`](../../../../course/solutions/python/tinyllm/xfmr/pos.py).

### 5.3 Why this solution

- **Cost.** Building the table is $O(\text{max\_len} \cdot d)$ once; each forward is a slice and an add, $O(Td)$. The learned table costs $L \cdot d$ parameters and its gradient touches only $T$ rows.
- **Interleaved vs. half layout.** fairseq and some HF code put all sines first, then all cosines. That is equally valid but needs a different rotation helper; interleaving matches the paper and `M00.2`, so the shift law reuses existing code.
- **Precompute vs. compute on the fly.** Computing rows per call would allow unbounded positions, but costs trig on every step; a fixed `max_len` matches the learned table's contract and makes "past the table" a clear error.
- **float64 then float32.** Computing in float32 is faster but wrong at long positions (section 2.1); the cost of float64 is paid once.
- **Tradeoff accepted.** Neither encoding extrapolates past `max_len`. The rotation property is the seed of RoPE (`L7.3`), which removes that limit for relative distances.

### 5.4 Explain it back

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

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. sin and cos concatenated (or swapped) instead of interleaved | the shift law fails; checkpoints from other code disagree | writes `pe[:, 0::2] = sin`, `pe[:, 1::2] = cos`; see 5.2 |
| 2. exponent $i/d$ instead of $2i/d$ | the ladder ends at $b^{-1/2}$: far positions alias | ratio `base ** (-2.0 / d)` over `d // 2` terms |
| 3. angles in float32 | rows past a few thousand off in the 4th decimal | angles built from a float64 `arange`, cast to float32 only at the end |
| 4. the shift turned the wrong way, or by the same angle for every pair | `shift_pe` goes back in position, or scrambles pairs | passes the per-pair vector `-k * omega` to `rotate_pairs` |
| 5. positions counted from 1 | every row one late; HF and torch disagree | `np.arange(offset, offset + T)` starts at 0 when `offset = 0` |
| 6. `offset` ignored | incremental decoding gives every new token position 0 | both modules slice rows `offset .. offset + T - 1` |
| 7. the fixed table registered as a parameter | it trains and lands in the checkpoint | `self.table` is a plain ndarray |
| 8. learned rows read as constants | `wpe` never learns | rows gathered with `F.embedding(self.weight, ...)` |
| `std` ignored by the learned table | every table starts at 0.02 | `normal_init((max_len, d), std, r)` uses the argument |
| an odd width accepted | the last feature has no partner | `_check_d` rejects odd or tiny `d` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M00.3` | `geometric(1, b^{-2/d}, d/2)` is the frequency ladder |
| Back | `M00.2` | `rotate_pairs` is the shift |
| Back | `M07.3` | `normal_init` for the learned table |
| Back | `M06.3` | `PCG32`, the init stream |
| Back | `L0.2` | `embedding` reads learned rows with their gradient |
| Back | `L0.1` | the `Tensor` the table is added to |
| Back | `L0.4` | `Module` for both encodings |
| Forward | `L14.3` | Whisper decoding calls this module through its declared contract. |
| Forward | `L5.5` | `SinusoidalPE` on both the source and the target embeddings |
| Forward | `L6.1` | `LearnedPE` is GPT-2's `wpe` |
| Forward | `L6.2` | `LearnedPE` is BERT's position embedding |
| Forward | `L7.3` | RoPE rotates queries and keys by the same ladder instead |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `sinusoidal_pe` | fairseq `SinusoidalPositionalEmbedding` | the half layout (all sin, then all cos) and a padding offset | `fairseq/modules/sinusoidal_positional_embedding.py` |
| `shift_pe` | RoPE | the rotation applied inside attention, so scores depend on $p - q$ only | `L7.3`; Su et al. 2021 |
| `LearnedPE` | HF `GPT2Model.wpe`, `BertEmbeddings.position_embeddings` | absolute positions capped at `n_positions` | `transformers/models/gpt2/modeling_gpt2.py` |
| no position at all | ALiBi, NoPE | a distance penalty on scores, or causality alone, for length extrapolation | Press et al. 2022; Kazemnejad et al. 2023 |
