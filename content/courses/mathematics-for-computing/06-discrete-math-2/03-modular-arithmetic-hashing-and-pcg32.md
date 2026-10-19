# Modular arithmetic, hashing, and PCG32 in Python

## Overview

| | |
|---|---|
| **Module** | `M06.3` · build · Python · Pass 2 · about 2 h reading |
| **Solution** | [`rng.py`](../../../../course/solutions/python/tinyllm/num/rng.py): `PCG32` (`next_u32`, `uniform`, `uniforms`, `below`, `shuffle`, `substream`, `state`, `set_state`), `splitmix64`, `child_seed`, `fnv1a64`, `universal_hash` |
| **Contract** | [`course/contracts/py/tinyllm/num/rng.pyi`](../../../../course/contracts/py/tinyllm/num/rng.pyi) · the algorithm, to the bit: [`spec/pcg32.md`](../../../../course/contracts/spec/pcg32.md) |
| **Needs** | no earlier module. Reading: `S-M05` (counting), `S-M06a` (modular arithmetic and hashing by hand) |
| **Used by** | `M07.0` normal draws by Box-Muller · `L0.2` dropout masks · `L0.4` the default init and dropout streams · `L0.5` the bigram's sampler · `L0.6` the token stream's window starts · later `L8.1` the sampler, `rt.04` KV block hashes, `data.03` Bloom hashing, `data.04` MinHash, and `ethics.04` reproducible bootstrap resampling; ported to Rust by `L10.1` and to Go by `load.01` · later: `L2.2`, `L2.3`, `L3.2`, `L3.3`, `L3.6`, `L4.1`, `L4.2`, `L4.3`, `L5.3`, `L5.4`, `L5.5`, `L6.1`, `L6.2`, `L6.3`, `L6.5`, `L6.7`, `L7.2`, `L7.5`, `L7.6`, `L7.8`, `L7.9` · `L14.5` the WER bootstrap stream (`PCG32(seed)`), `L13.1`, `L13.3`, `L13.5`, `L14.2` · `L14.3` Whisper decoding · `L13.6` image preprocessing; `L14.4` |
| **Optional depth** | O'Neill, "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation" (2014); Knuth, *TAOCP* vol. 2, ch. 3 (linear congruential generators, the spectral test); Steele, Lea, and Flood, "Fast Splittable Pseudorandom Number Generators" (2014); Carter and Wegman, "Universal Classes of Hash Functions" (1979); Noll's FNV page (isthe.com/chongo/tech/comp/fnv) |

## Key Takeaways

- Python integers are unbounded, so PCG's 64-bit state transition explicitly applies `& (2**64 - 1)` after each multiply and add.
- **PCG32** is a 64-bit linear congruential generator whose state is hidden behind a permutation (an xorshift and a rotation chosen by the top bits); seeded the same way, Python, Rust, and Go produce the same stream bit for bit.
- A **uniform double** with all 53 bits takes two 32-bit draws and only integer arithmetic, so it is bit-exact across languages and never equals 1.0.
- `r mod n` is **biased** unless $n$ divides $2^{32}$; rejection removes the bias.
- **Sub-streams** derived with SplitMix64 keep initialization, dropout, shuffling, and sampling independent of each other; **FNV-1a** chains over buffers, which is how a KV block hash extends its parent's.

## How to read this chapter

Sections 1 to 3 build modular arithmetic, hashing, and the PCG32 generator from first principles and compute the first outputs by hand. Section 4 fixes the interface and the cases a correct solution must handle, most of them published vectors. Section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

From Pass 2 on, the system makes random choices everywhere: initial weights, dropout masks, the order of training windows, the token the sampler picks. Every one of them has to be reproducible from one seed (P11), or a failing run cannot be replayed, a resumed checkpoint diverges from the run it continues, and a parity test between the Python sampler and the Rust engine (`L10.1`) has nothing to compare. numpy's generator, Rust's `rand`, and Go's `math/rand` all produce different streams for the same seed. So the course fixes one generator to the bit, PCG32 (`spec/pcg32.md`), and this module implements it in Python for training. Published vectors give the later Rust and Go ports the same oracle. The same mathematics, arithmetic modulo a power of two, gives the hash functions the system needs: FNV-1a for KV block hashes and universal hashing for hash tables.

## 2. Principles

| Symbol | Meaning | Type |
|---|---|---|
| $a \bmod n$ | the remainder of $a$ divided by $n$, in $\{0, \dots, n-1\}$ | integer |
| $\mathbb{Z}_n$ | the integers $\{0, \dots, n-1\}$ with $+$ and $\times$ taken mod $n$ | |
| $s_t$ | generator state after $t$ steps | `u64` |
| $A$ | the PCG multiplier 6364136223846793005 | `u64` |
| $c$ | the increment, `inc` $= 2 \cdot \mathit{seq} + 1$, always odd | `u64` |
| $\mathrm{rotr}_{32}(x, k)$ | rotate the 32-bit $x$ right by $k$ bits | `u32` |
| $u$ | a uniform draw in $[0, 1)$ | `double` |
| $h, m, p$ | a hash function, its number of buckets, a prime | |
| $\oplus$ | bitwise exclusive or (`^`) | |

### 2.1 Modular arithmetic

Every integer $a$ and modulus $n \ge 1$ give a unique quotient and remainder, $a = qn + r$ with $0 \le r < n$; write $r = a \bmod n$. Two integers are **congruent** mod $n$ when they have the same remainder. Remainders respect addition and multiplication: $(a + b) \bmod n = ((a \bmod n) + (b \bmod n)) \bmod n$, and the same for $\times$. So one may reduce at every step instead of at the end, and $\mathbb{Z}_n$ is a closed number system.

A `uint64_t` in C holds exactly $\mathbb{Z}_{2^{64}}$: the C standard defines unsigned overflow as wraparound, which *is* reduction mod $2^{64}$. Python integers never overflow, so `x * y` of two 64-bit values has up to 128 bits; the solution reduces with `& (2**64 - 1)` (keeping the low 64 bits is the remainder mod $2^{64}$). Forget it once and the state grows without bound while the outputs, which read the low bits, go quietly wrong after the first wrap.

Two facts about $\mathbb{Z}_{2^k}$ carry the rest of the chapter. An odd number has a multiplicative inverse mod $2^k$, so multiplying by an odd constant is a **bijection** (it scrambles without losing information). And $x \mapsto x \oplus (x \gg k)$ is a bijection too (the top $k$ bits pass through unchanged and determine how to undo the rest).

### 2.2 Hashing

A **hash function** maps keys to bucket numbers, $h: U \to \{0, \dots, m-1\}$. Two keys **collide** when $h(x) = h(y)$. With $n$ keys in $m$ buckets the **load factor** is $\alpha = n/m$, and if collisions are as rare as for random assignment, a lookup scans about $1 + \alpha$ keys.

No single fixed function is good for every set of keys (an adversary can pick keys that all collide). A **universal family** picks $h$ at random so that for any two different keys, $\Pr[h(x) = h(y)] \le 1/m$. Carter and Wegman's family: choose a prime $p$ larger than every key, draw $a \in \{1, \dots, p-1\}$ and $b \in \{0, \dots, p-1\}$, and set

$$h_{a,b}(x) = ((a x + b) \bmod p) \bmod m .$$

Why it works: for $x \ne y$, the map $(a, b) \mapsto (r, s) = (ax + b \bmod p,\ ay + b \bmod p)$ is a bijection onto the pairs with $r \ne s$, because $x - y$ has an inverse mod the prime $p$, so $a = (r - s)(x - y)^{-1}$ and then $b$ are determined. A collision needs $r \equiv s \pmod m$; for each $r$, at most $\lceil p/m \rceil - 1 \le (p-1)/m$ values $s \ne r$ qualify. That gives at most $p(p-1)/m$ colliding pairs out of $p(p-1)$, a probability of at most $1/m$. The order of the two reductions matters: $((ax + b) \bmod m) \bmod p$ is a different function and loses the guarantee.

**FNV-1a** is a fast non-cryptographic hash of a byte string: start from the 64-bit offset basis $h_0 = \texttt{0xCBF29CE484222325}$ and for each byte $b$ do $h \leftarrow (h \oplus b) \cdot P \bmod 2^{64}$ with the prime $P = \texttt{0x100000001B3}$. Xor first, then multiply: that order is the "1a" (FNV-1 multiplies first, and hashes differently). Because the state between bytes is just $h$, hashing a buffer in two pieces, the second starting from the first's result, equals hashing it at once: $\mathrm{fnv}(b, \mathrm{fnv}(a)) = \mathrm{fnv}(a \| b)$. `rt.04` uses exactly that to chain each KV block's hash onto its parent's.

### 2.3 From an LCG to PCG32

A **linear congruential generator** (LCG) steps $s_{t+1} = (A s_t + c) \bmod 2^{64}$. With $c$ odd and $A \equiv 1 \pmod 4$ it visits all $2^{64}$ states before repeating (the Hull-Dobell theorem). Its weakness is the low bits: bit $k$ of $s_t$ depends only on bits $0..k$ of the previous state, so it repeats with period $2^{k+1}$, and the lowest bit simply alternates. An LCG's raw low bits are not random at all.

PCG keeps the LCG for the state and outputs a **permutation** of the *old* state that only uses its good high bits:

```
next_u32():
    old   = state
    state = old * A + inc                          (mod 2^64)
    xs    = u32(((old >> 18) XOR old) >> 27)       (xorshift: high bits fold into the middle)
    rot   = old >> 59                              (the top 5 bits, 0 to 31)
    return rotr32(xs, rot)                         (a rotation chosen by the state itself)
```

Taking the output from `old` lets C compute the multiply and the permutation in parallel. `inc` selects one of $2^{63}$ **streams**: `inc = (seq << 1) | 1`, forced odd, because an even increment breaks the full period. **Seeding** is O'Neill's `pcg32_srandom_r(seed, seq)`: `state = 0`, one step, `state += seed`, one step. The two steps push the seed through the multiplier so that seeds 0 and 1 do not give nearly identical first outputs.

In C, the rotation has one trap: `xs << (32 - rot)` with `rot = 0` shifts a 32-bit value by 32, which is **undefined behavior** (x86 and ARM happen to compute something, the optimizer may compute something else, and UBSan stops the program). Write the left shift as `xs << ((0u - rot) & 31u)`: for `rot = 0` it shifts by 0, and `xs | xs` is still `xs`.

### 2.4 Uniform doubles from integers

A double has a 53-bit significand, so the finest grid of evenly spaced doubles in $[0, 1)$ is $k \cdot 2^{-53}$ for $k = 0, \dots, 2^{53} - 1$. One 32-bit draw is not enough bits; the spec uses two:

$$u = \left(\lfloor a / 2^5 \rfloor \cdot 2^{26} + \lfloor b / 2^6 \rfloor\right) \cdot 2^{-53},$$

27 bits from $a$ and 26 from $b$. The integer in the parentheses is below $2^{53}$, so the double holds it exactly, and multiplying by a power of two is exact too. No rounding happens anywhere, which is why `uniform()` is bit-identical in every language, and why $u < 1$ always: inverse-CDF sampling (`L8.1`) relies on never seeing 1.0. In C, compute in `uint64_t`: `(a << 26)` on a 32-bit type loses the top bits.

### 2.5 Unbiased integers and shuffling

To draw an integer in $[0, n)$ from $r$ uniform on $[0, 2^{32})$, `r % n` is biased whenever $n$ does not divide $2^{32}$: the first $2^{32} \bmod n$ residues get one extra preimage. For $n = 3 \cdot 2^{30}$ the values below $2^{30}$ come up half the time instead of a third. The fix is **rejection**: let $t = 2^{32} \bmod n$ (computed as `(2**32 - n) % n` so it fits in 32 bits), draw until $r \ge t$, return $r \bmod n$. The accepted range has $2^{32} - t$ values, a multiple of $n$, so every residue is equally likely; fewer than half of the draws are ever rejected.

**Fisher-Yates** shuffles a list uniformly: for $i = n-1$ down to $1$, pick $j$ uniformly in $[0, i]$ and swap positions $i$ and $j$. Each of the $n!$ sequences of choices gives a different permutation, so all are equally likely. Picking $j$ in $[0, i)$ instead (Sattolo's algorithm) never leaves an element in place and produces only the $(n-1)!$ cyclic permutations.

### 2.6 Sub-streams with SplitMix64

One user seed must feed several independent consumers (initialization, dropout, shuffling, sampling), so that turning dropout on does not change the initial weights. The spec derives each purpose's generator from the seed alone:

$$\mathrm{child\_seed}(s, p) = \mathrm{mix64}(s + p \cdot \texttt{0x9E3779B97F4A7C15}), \qquad \mathrm{stream}(s, \mathit{purpose}) = \mathrm{PCG32}(\mathrm{child\_seed}(s, p), \mathit{seq} = p),$$

with purpose ids init 1, dropout 2, shuffle 3, sample 4, mutation 5. `mix64` is SplitMix64's output function, two rounds of xorshift and odd multiply (both bijections, 2.1):

```
z = (z XOR (z >> 30)) * 0xBF58476D1CE4E5B9
z = (z XOR (z >> 27)) * 0x94D049BB133111EB
return z XOR (z >> 31)                                          (all mod 2^64)
```

A SplitMix64 generator seeded with $s$ adds the golden-ratio constant to its state and mixes, so its $k$-th output is $\mathrm{mix64}(s + k \cdot \texttt{0x9E37...})$: the child seed of purpose $p$ is simply output $p$ of SplitMix64 seeded with $s$. It depends on the seed, never on how many draws the parent made.

## 3. Worked example by hand

**The first output of `pcg32(0)`** (the spec's `worked_example_seed_0`). Seeding with `seed = 0`, `seq = 54`:

| Step | Computation | Result |
|---|---|---|
| inc | $(54 \ll 1) \mid 1 = 108 + 1$ | 109 |
| first step | $0 \cdot A + 109$ | state = 109 |
| add the seed | $109 + 0$ | 109 |
| second step | $109 \cdot A + 109 \bmod 2^{64}$ | state = `0x9AE4F7499BA72696` |

Now `next_u32()` with `old = 0x9AE4F7499BA72696`:

| Quantity | Value |
|---|---|
| `old >> 18` | `0x000026B93DD266E9` |
| `(old >> 18) XOR old` | `0x9AE4D1F0A675407F` |
| `... >> 27`, low 32 bits | `xs = 0x5C9A3E14` |
| `rot = old >> 59` | 19 (the top 5 bits, `10011`) |
| `rotr32(xs, 19)` = `(xs >> 19) \| (xs << 13)` (low 32 bits) | **`0x47C28B93`** = 1203932051 |

That is the first entry of `next_u32["0"]` in `spec/pcg32.vectors.json`. The second draw is `0xB98F6A27`, so the first `uniform()` is $\lfloor \texttt{0x47C28B93} / 32 \rfloor = 37622876$ and $\lfloor \texttt{0xB98F6A27} / 64 \rfloor = 48643496$, giving $(37622876 \cdot 2^{26} + 48643496) \cdot 2^{-53} = 2524828517416360 \cdot 2^{-53} = 0.2803122753265841$.

**FNV-1a of `"a"`**: `"a"` is the byte `0x61`. $h = \texttt{0xCBF29CE484222325} \oplus \texttt{0x61} = \texttt{0xCBF29CE484222344}$, then $h \cdot \texttt{0x100000001B3} \bmod 2^{64} = \texttt{0xAF63DC4C8601EC8C}$ (the full product is `0xCBF29CE5DEAF63DC4C8601EC8C`; keep the low 16 hex digits).

**A universal hash**: $a = 3$, $b = 7$, $p = 13$, $m = 5$, $x = 10$. $3 \cdot 10 + 7 = 37$; $37 \bmod 13 = 11$; $11 \bmod 5 = \mathbf{1}$.

## 4. The interface

```python
# tinyllm/num/rng.py (contract: rng.pyi)
class PCG32:
    def __init__(self, seed: int, seq: int = 54) -> None: ...
    def next_u32(self) -> int: ...
    def uniform(self) -> float: ...
    def uniforms(self, n: int) -> NDArray: ...
    def below(self, n: int) -> int: ...
    def shuffle(self, xs: MutableSequence[T]) -> None: ...
    def substream(self, purpose: str) -> "PCG32": ...
    def state(self) -> tuple[int, int]: ...
    def set_state(self, s: tuple[int, int]) -> None: ...
def splitmix64(x: int) -> int: ...
def child_seed(seed: int, purpose_id: int) -> int: ...
def fnv1a64(data: bytes, h: int = 0xCBF29CE484222325) -> int: ...
def universal_hash(x: int, a: int, b: int, p: int, m: int) -> int: ...
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Seed 0 by hand | unit | section 3: the seeded state and `0x47C28B93` | pins the seeding procedure |
| O'Neill's demo line | golden | `PCG32(42)` gives O'Neill's published first six outputs | an oracle outside the course |
| 1024 outputs | golden | 1024 outputs for seeds 0, 1, $2^{63}$ | every language's stream |
| Bit-exact uniforms | golden | the spec's `uniform_f64` vectors, also through `uniforms` | the sampler's one draw per token |
| 53 bits, below 1 | property | $u \cdot 2^{53}$ integral, $u < 1$, more than 32 bits used | inverse-CDF sampling in `L8.1` |
| `below` and `shuffle` vectors | golden | `below(10)` and `shuffle([0..9])` vectors | `L0.5` replays its data order on resume |
| No modulo bias | statistical | $n = 3 \cdot 2^{30}$: the share below $2^{30}$ is 1/3, not 1/2 | unbiased sampling of large ranges |
| Bad `n` for `below` | boundary | $n = 0$, negative, or above $2^{32}$ raise | |
| Sub-stream vectors | golden | `child_seed` and the first outputs of each purpose | init, dropout, shuffle, sample stay independent |
| Sub-stream independence | property | a sub-stream depends on the seed only | adding dropout does not change init |
| SplitMix64 outputs | golden | SplitMix64 seeded 0 and 1234567, published outputs | the mixer is the standard one |
| FNV-1a vectors | golden | FNV-1a 64 reference vectors | the KV block hash in `rt.04` and the Rust engine |
| FNV-1a chaining | property | `fnv(b, fnv(a)) == fnv(a + b)` on random bytes | chained block hashes |
| Universal hash by hand | unit | section 3's universal hash | |
| Universal collision rate | property | exhaustive over all $(a, b)$ for $p = 101$: collisions at most $1/m + 1/p$ | hash tables in `ds.*` |
| State round trip | property | `set_state(state())` continues the stream; an even `inc` is rejected | checkpoints store the generator (`trainer_state.json`) |
| Negative `n` for `uniforms` | boundary | `uniforms(-1)` raises, `uniforms(0)` is empty | |

## 5. The solution, step by step

### From first principles

1. Every random choice must replay from one seed in Python, Rust, and Go (section 1). So the generator cannot be a library's; it must be a fixed algorithm specified to the bit, with published vectors: PCG32 as in [`spec/pcg32.md`](../../../../course/contracts/spec/pcg32.md).
2. The state lives in $\mathbb{Z}_{2^{64}}$ and outputs in $\mathbb{Z}_{2^{32}}$ (2.1). C gets the reduction from `uint64_t` overflow; Python ints never overflow, so every multiply and add that can exceed 64 bits is followed by `& MASK64`, and every 32-bit output by `& MASK32`.
3. The raw LCG has weak low bits (2.3), so the output is a permutation of the *old* state built from its high bits: an xorshift, then a rotation by the top 5 bits. The increment is forced odd, and seeding follows O'Neill's two-step procedure exactly, or no vector matches.
4. Anything built on top must also be bit-exact. A uniform double therefore uses only integer arithmetic and one exact scaling by $2^{-53}$ (2.4); a bounded integer uses rejection, which is exact and unbiased (2.5); the shuffle is Fisher-Yates with $j \in [0, i]$.
5. Independent purposes need independent generators that depend only on the seed (2.6). So a sub-stream is a fresh `PCG32` seeded by `child_seed(seed, purpose)`, never derived from the parent's current state.
6. The hashes are the same arithmetic: FNV-1a is xor then multiply mod $2^{64}$, taking an optional starting value so buffers chain; the universal hash reduces mod $p$ before mod $m$ (2.2).

### The code, annotated

The constants are the spec's, named once:

```python
MASK64 = (1 << 64) - 1          # x & MASK64 == x mod 2^64
MASK32 = (1 << 32) - 1
MULT = 6364136223846793005      # the PCG multiplier A (odd, A = 1 mod 4)
GOLDEN = 0x9E3779B97F4A7C15     # SplitMix64's increment
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
PURPOSES = {"init": 1, "dropout": 2, "shuffle": 3, "sample": 4, "mutation": 5}
```

Seeding is `pcg32_srandom_r(seed, seq)`: state 0, one step, add the seed, one step. Section 3 traces it for seed 0, landing on state `0x9AE4F7499BA72696`:

```python
class PCG32:
    def __init__(self, seed: int, seq: int = 54) -> None:
        self.seed = seed & MASK64       # kept: substream derives children from it
        self._state = 0
        # inc must be odd: an LCG mod 2^64 has full period only with an odd
        # increment. seq picks one of 2^63 distinct streams.
        self._inc = ((seq << 1) | 1) & MASK64
        self.next_u32()                 # state = inc
        self._state = (self._state + self.seed) & MASK64
        self.next_u32()                 # push the seed through the multiplier
```

`next_u32` advances the state and permutes the old one. The rotation is written with `(-rot) & 31` as the left-shift amount, the same expression a C port needs to avoid shifting by 32 when `rot = 0`:

```python
    def next_u32(self) -> int:
        old = self._state
        self._state = (old * MULT + self._inc) & MASK64   # reduce: Python ints grow
        # Output from the OLD state: xorshift high bits down, keep 32 bits...
        xs = (((old >> 18) ^ old) >> 27) & MASK32
        rot = old >> 59                                   # ...rotate by the top 5 bits
        return ((xs >> rot) | (xs << ((-rot) & 31))) & MASK32
```

`uniform` builds a 53-bit integer from 27 + 26 bits and scales it by $2^{-53}$. Both steps are exact in a double, so every language gets the same bits and the result is at most $1 - 2^{-53}$:

```python
    def uniform(self) -> float:
        a = self.next_u32() >> 5  # 27 bits
        b = self.next_u32() >> 6  # 26 bits
        # a * 2^26 + b is an integer below 2^53, exact in a double, and the
        # product with 2^-53 is exact too: no rounding anywhere.
        return (a * 67108864 + b) * (1.0 / 9007199254740992.0)
```

`below` computes the rejection threshold $t = 2^{32} \bmod n$ as `(2**32 - n) % n`, which is the same residue, and redraws until $r \ge t$. `shuffle` is Fisher-Yates from the top down:

```python
    def below(self, n: int) -> int:
        if not 1 <= n <= 1 << 32:
            raise ValueError(f"below: n must be in [1, 2^32], got {n}")
        # 2^32 mod n raw values would map to the low residues one extra
        # time; rejecting r < t leaves a multiple of n values, each residue
        # equally often.
        t = ((1 << 32) - n) % n
        while True:
            r = self.next_u32()
            if r >= t:
                return r % n

    def shuffle(self, xs: MutableSequence[T]) -> None:
        for i in range(len(xs) - 1, 0, -1):
            j = self.below(i + 1)  # 0 <= j <= i: xs[i] may stay put
            xs[i], xs[j] = xs[j], xs[i]
```

A sub-stream is a new generator whose seed and stream id depend only on the parent's *seed* and the purpose, so draws already made on the parent cannot change it. `state` and `set_state` expose exactly the two words a checkpoint needs, and `set_state` refuses an even increment, which no PCG32 can have:

```python
    def substream(self, purpose: str) -> "PCG32":
        p = PURPOSES[purpose]
        return PCG32(child_seed(self.seed, p), seq=p)

    def state(self) -> tuple[int, int]:
        return (self._state, self._inc)

    def set_state(self, s: tuple[int, int]) -> None:
        state, inc = int(s[0]), int(s[1])
        if inc % 2 == 0:     # a corrupt checkpoint, not a PCG32 state
            raise ValueError(f"set_state: inc must be odd, got {inc}")
        self._state = state & MASK64
        self._inc = inc & MASK64
```

SplitMix64's mixer is two xorshift-multiply rounds and a final xorshift, each a bijection of 64-bit words, and `child_seed` is SplitMix64's output number `purpose_id` for that seed:

```python
def splitmix64(x: int) -> int:
    z = x & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def child_seed(seed: int, purpose_id: int) -> int:
    return splitmix64((seed + purpose_id * GOLDEN) & MASK64)
```

The two hashes. `fnv1a64` takes the previous hash as its starting value, which is what makes `fnv1a64(b, fnv1a64(a)) == fnv1a64(a + b)`; `universal_hash` validates the family's parameters and reduces in the order that carries the $1/m$ guarantee:

```python
def fnv1a64(data: bytes, h: int = FNV_OFFSET) -> int:
    h &= MASK64
    for byte in bytes(data):
        h ^= byte  # xor first, then multiply: that order is the "1a"
        h = (h * FNV_PRIME) & MASK64
    return h


def universal_hash(x: int, a: int, b: int, p: int, m: int) -> int:
    # ... ValueError unless 1 <= a < p, 0 <= b < p, and m >= 1
    return ((a * x + b) % p) % m   # mod the prime first, then into m buckets
```

Full solution: [`rng.py`](../../../../course/solutions/python/tinyllm/num/rng.py).

### Why this solution

- **Cost.** `next_u32` is a constant number of big-int operations on at most 128-bit values; `uniform` is two draws; `below` needs fewer than two draws on average (the rejected fraction is $t / 2^{32} < 1/2$); `shuffle` is $O(n)$ draws; `fnv1a64` is $O(\text{bytes})$. Python makes all of this slow next to C, but training draws one number per weight or per mask element, which is fine at this scale; `L10.1` ports the same code to Rust for the engine.
- **Why not numpy's or Python's generator.** They are excellent generators, but their streams are not specified across languages, so a Python sampler and a Rust engine could never agree token for token, and a checkpoint could not store the generator in a portable form.
- **Why PCG32 and not a raw LCG or xorshift.** A raw LCG's low bits have tiny periods (2.3); PCG hides them behind the permutation at almost no cost. xorshift generators fail some statistical tests on their low bits and have no cheap stream selection. PCG32 has small state (two words), streams, and published reference outputs.
- **Why rejection and not `r % n` or floating-point scaling.** `r % n` is biased (2.5). `floor(uniform() * n)` is also biased for large $n$ and depends on floating-point rounding. Rejection is exact and portable. Lemire's multiply-shift method is faster but would change the stream, and the spec fixes this one.
- **Why sub-streams from the seed.** Deriving a child from the parent's current state would tie dropout to the number of initialization draws, so changing the model size would change the dropout masks.
- **Tradeoff accepted.** Every arithmetic step carries an explicit mask. It is noise to read, but it is exactly the reduction modulo $2^{64}$ the mathematics requires, and dropping one mask is the bug that matches for a while and then diverges.

### Explain it back

1. Why does `next_u32` compute the output from `old` and not from the new state?

   <details><summary>Answer</summary>

   Because the spec says so: PCG-XSH-RR permutes the state *before* the step. A version that permutes the new state is an equally plausible generator that matches no published vector and no other port. (The design reason is speed in C: the multiply and the permutation of the old state can run in parallel.)

   </details>

2. What goes wrong if one `& MASK64` is dropped in `next_u32`?

   <details><summary>Answer</summary>

   Until the state first exceeds $2^{64}$ the outputs are identical, because the permutation reads only bits below 64 at first. After that the state is a growing big integer: `old >> 59` is no longer the top 5 bits of a 64-bit word, so the rotation and then everything else is wrong, forever, while the code raises nothing.

   </details>

3. Why is `((1 << 32) - n) % n` the right threshold, and why is a value `r < t` rejected rather than `r >= 2**32 - t`?

   <details><summary>Answer</summary>

   $(2^{32} - n) \bmod n = 2^{32} \bmod n$, computed without a number above $2^{32}$. Rejecting the $t$ smallest values leaves $2^{32} - t$ values, a multiple of $n$, so each residue has the same number of preimages. Rejecting the $t$ largest values would also be unbiased, but it is a different stream; the spec fixes the low end.

   </details>

4. Why does `shuffle` call `below(i + 1)` and not `below(i)`?

   <details><summary>Answer</summary>

   Fisher-Yates picks $j \in [0, i]$, so that element $i$ may stay where it is. With $j \in [0, i)$ (Sattolo) no element ever stays put and only the $(n-1)!$ cyclic permutations are produced, not all $n!$.

   </details>

5. Why does `substream` use `self.seed` and not `self._state`?

   <details><summary>Answer</summary>

   The seed never changes, while the state moves with every draw. Using the seed makes `PCG32(s).substream("dropout")` the same generator however many numbers the parent produced, so adding dropout, or changing how many weights are initialized, leaves every other purpose's stream unchanged.

   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| permuting the new state instead of the old one | a plausible stream that matches no other language | saves `old` before the step and permutes it |
| an even increment (`seq << 1` without `\| 1`) | short period; wrong from the first output | `inc = ((seq << 1) \| 1) & MASK64` |
| no `& MASK64` in Python | correct until the state first wraps, then wrong forever | masks after every multiply and add; see step 2 |
| a uniform from one 32-bit draw | only 32 random bits, so the lower 21 bits of a double are always zero | two draws, 27 + 26 bits, scaled by $2^{-53}$ |
| seeding in the wrong order (seed before the first step) | streams for nearby seeds look alike, and no vector matches | step, add the seed, step, as in `pcg32_srandom_r` |
| `j = below(i)` in the shuffle (Sattolo) | only cyclic permutations; no element ever stays put | `j = self.below(i + 1)` |
| `r % n` without rejection | small values favoured: 1/2 instead of 1/3 for $n = 3 \cdot 2^{30}$ | rejects `r < t` with `t = (2**32 - n) % n` |
| deriving a sub-stream from the current state | dropout masks change when the number of init draws changes | seeds the child from `self.seed`, never from `_state` |
| a different mixer (Murmur's `>> 33`) | child seeds differ from every other port | SplitMix64's shifts 30, 27, 31 and its two constants |
| multiply before xor (FNV-1, not FNV-1a) | every KV block hash differs from the Rust engine's | `h ^= byte` before `h *= FNV_PRIME` |
| reducing mod $m$ before mod $p$ | not universal; clustered buckets | `((a * x + b) % p) % m` |
| accepting an even `inc` in `set_state` | a corrupt checkpoint resumes a non-PCG sequence | `set_state` raises on an even `inc` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M05` | counting arguments behind the rejection threshold and $n!$ shuffles (reading) |
| Back | `S-M06a` | modular arithmetic, hashing, and load factor by hand (reading) |
| Forward | `M07.0` | `normal(rng, n)`: Box-Muller over two `uniform()` draws |
| Forward | `L0.2` | dropout masks and `gradcheck_all`'s random inputs |
| Forward | `L0.4` | layers default to `PCG32(0).substream(purpose)`, so initialization and dropout use separate streams and adding dropout does not change the initial weights |
| Forward | `L0.5` | the autograd bigram samples text with one `uniform()` per token |
| Forward | `L0.6` | `TokenStream` draws its window starts from a PCG32 and saves the generator's state in its cursor, so a resumed run reads the same batches |
| Forward | `ethics.04` | bias-evaluation bootstrap intervals resample paired prompts with PCG32 so the reports are reproducible |
| Forward | `L8.1` | the sampler: one `uniform()` per token on `stream(seed, sample)` (`spec/sampling.md`) |
| Forward | `L10.1`, `load.01` | the Rust and Go ports, held to the same published vectors |
| Forward | `data.03` | the Python Bloom screen derives stable bit positions from FNV-1a and SplitMix64 |
| Forward | `L14.5` | `bootstrap_ci` in the multimodal metrics draws its resamples from `PCG32(seed)` |
| Forward | `L13.1` | conv weights draw from a `PCG32` stream |
| Forward | `L13.3` | patch, position, and block weights draw from `PCG32` |
| Forward | `L13.5` | the dual encoder initializes from `PCG32` |
| Forward | `L14.2` | the encoder initializes from `PCG32` |
| Forward | `L14.3` | Whisper decoding calls this module through its declared contract. |
| Forward | `L13.6` | Image preprocessing uses this module directly. |
| Forward | `L2.2` | uses this module |
| Forward | `L2.3` | uses this module |
| Forward | `L3.2` | uses this module |
| Forward | `L3.3` | uses this module |
| Forward | `L3.6` | uses this module |
| Forward | `L4.1` | uses this module |
| Forward | `L4.2` | uses this module |
| Forward | `L4.3` | uses this module |
| Forward | `L5.3` | uses this module |
| Forward | `L5.4` | uses this module |
| Forward | `L5.5` | uses this module |
| Forward | `L6.1` | uses this module |
| Forward | `L6.2` | uses this module |
| Forward | `L6.3` | uses this module |
| Forward | `L6.5` | uses this module |
| Forward | `L6.7` | uses this module |
| Forward | `L7.2` | uses this module |
| Forward | `L7.5` | uses this module |
| Forward | `L7.6` | uses this module |
| Forward | `L7.8` | uses this module |
| Forward | `L7.9` | uses this module |
| Forward | `data.04` | uses this module |
| Forward | `L14.4` | Whisper decoding consumes this implementation. |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| PCG32 | the PCG family (`pcg64`, numpy's default `PCG64`) | 128-bit state, the DXSM output function, jump-ahead in $O(\log n)$ | numpy `numpy/random/_pcg64.pyx`; O'Neill's `pcg-cpp` |
| sub-streams by SplitMix64 | JAX's counter-based `threefry` keys | split a key into independent keys without any shared state, ideal for parallel devices | `jax/_src/prng.py` |
| `below` by rejection | Lemire's nearly divisionless method | one multiply and rarely a division instead of a modulo per draw | Lemire, "Fast Random Integer Generation in an Interval" (2019) |
| FNV-1a | xxHash, wyhash | many bytes per step with SIMD, better avalanche, still non-cryptographic | `xxhash.h` (`XXH3`) |
| block hash chaining | vLLM prefix caching | hashes each KV block with its parent hash and extra keys (LoRA id, multimodal inputs) | vLLM `vllm/v1/core/kv_cache_utils.py`, `hash_block_tokens` |

