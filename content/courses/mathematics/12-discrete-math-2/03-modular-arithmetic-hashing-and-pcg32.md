# Modular arithmetic, hashing, and PCG32 in Python

Once you start training models, the system makes random choices everywhere: initial weights, dropout masks, the order of training windows, the token the sampler picks. Every one of them has to be reproducible from one seed. Otherwise a failing run cannot be replayed, a resumed checkpoint diverges from the run it continues, and a parity test between the Python sampler and the Rust engine of [Candle model runner, Rust KV cache, sampler and PCG32](../../ai-engineering/23-model-serving/02-model-runner-and-sampler.md) has nothing to compare.

numpy's generator, Rust's `rand`, and Go's `math/rand` all produce different streams for the same seed. So the books fix one generator to the bit, PCG32, specified in [`spec/pcg32.md`](../../../../course/contracts/spec/pcg32.md), and this section implements it in Python for training. Published vectors give the later Rust and Go ports the same oracle. The same mathematics, arithmetic modulo a power of two, also gives the system the hash functions it needs: FNV-1a for KV block hashes and universal hashing for hash tables.

The solution is [`rng.py`](../../../../course/solutions/python/tinyllm/num/rng.py), which provides `PCG32` (`next_u32`, `uniform`, `uniforms`, `below`, `shuffle`, `substream`, `state`, `set_state`) together with `splitmix64`, `child_seed`, `fnv1a64`, and `universal_hash`; its contract is [`rng.pyi`](../../../../course/contracts/py/tinyllm/num/rng.pyi). No earlier code is needed. The counting arguments come from the [Discrete Math 1 problem set](../11-discrete-math-1/03-problem-set/README.md), and if you want to practise modular arithmetic and hashing by hand first, work through [the first problem set of this chapter](04-problem-set-a.md).

> **Learning objectives**
> - Compute in $\mathbb{Z}_{2^{64}}$ in Python, where integers never overflow, and explain why every step needs an explicit mask.
> - Derive PCG32 from a linear congruential generator, and compute its first output for seed 0 by hand.
> - Build a uniform double with all 53 bits from integer arithmetic alone, and draw unbiased bounded integers by rejection.
> - Prove that the Carter-Wegman family is universal, and explain why FNV-1a chains over buffers.
> - Derive independent sub-streams from one seed with SplitMix64 so that one consumer's draws never change another's.

## Modular arithmetic

The notation for the section is gathered here first.

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

Every integer $a$ and modulus $n \ge 1$ give a unique quotient and remainder, $a = qn + r$ with $0 \le r < n$; write $r = a \bmod n$. Two integers are **congruent** mod $n$ when they have the same remainder. Remainders respect addition and multiplication: $(a + b) \bmod n = ((a \bmod n) + (b \bmod n)) \bmod n$, and the same holds for $\times$. So you may reduce at every step instead of at the end, and $\mathbb{Z}_n$ is a closed number system.

A `uint64_t` in C holds exactly $\mathbb{Z}_{2^{64}}$: the C standard defines unsigned overflow as wraparound, which *is* reduction mod $2^{64}$. Python integers never overflow, so `x * y` of two 64-bit values has up to 128 bits. The solution reduces with `& (2**64 - 1)`, since keeping the low 64 bits is the remainder mod $2^{64}$. Forget it once and the state grows without bound while the outputs, which read the low bits, go quietly wrong after the first wrap.

Two facts about $\mathbb{Z}_{2^k}$ carry the rest of the section. An odd number has a multiplicative inverse mod $2^k$, so multiplying by an odd constant is a **bijection**: it scrambles without losing information. And $x \mapsto x \oplus (x \gg k)$ is a bijection too, because the top $k$ bits pass through unchanged and determine how to undo the rest.

## Hashing

The first use of this arithmetic is hashing. A **hash function** maps keys to bucket numbers, $h: U \to \{0, \dots, m-1\}$. Two keys **collide** when $h(x) = h(y)$. With $n$ keys in $m$ buckets the **load factor** is $\alpha = n/m$, and if collisions are as rare as for random assignment, a lookup scans about $1 + \alpha$ keys.

No single fixed function is good for every set of keys, since an adversary can pick keys that all collide. A **universal family** instead picks $h$ at random so that for any two different keys, $\Pr[h(x) = h(y)] \le 1/m$. Carter and Wegman's family works like this: choose a prime $p$ larger than every key, draw $a \in \{1, \dots, p-1\}$ and $b \in \{0, \dots, p-1\}$, and set

$$h_{a,b}(x) = ((a x + b) \bmod p) \bmod m .$$

Here is why it works. For $x \ne y$, the map $(a, b) \mapsto (r, s) = (ax + b \bmod p,\ ay + b \bmod p)$ is a bijection onto the pairs with $r \ne s$, because $x - y$ has an inverse mod the prime $p$, so $a = (r - s)(x - y)^{-1}$ and then $b$ are determined. A collision needs $r \equiv s \pmod m$; for each $r$, at most $\lceil p/m \rceil - 1 \le (p-1)/m$ values $s \ne r$ qualify. That gives at most $p(p-1)/m$ colliding pairs out of $p(p-1)$, a probability of at most $1/m$. The order of the two reductions matters: $((ax + b) \bmod m) \bmod p$ is a different function and loses the guarantee.

**FNV-1a** is a fast non-cryptographic hash of a byte string. Start from the 64-bit offset basis $h_0 = \texttt{0xCBF29CE484222325}$, and for each byte $b$ do $h \leftarrow (h \oplus b) \cdot P \bmod 2^{64}$ with the prime $P = \texttt{0x100000001B3}$. Xor first, then multiply: that order is the "1a" (FNV-1 multiplies first, and hashes differently). Because the state between bytes is just $h$, hashing a buffer in two pieces, the second starting from the first's result, equals hashing it at once: $\mathrm{fnv}(b, \mathrm{fnv}(a)) = \mathrm{fnv}(a \| b)$. The [Paged KV block pool](../../ai-engineering/21-inference/08-paged-kv-block-pool.md) uses exactly that to chain each KV block's hash onto its parent's.

## From an LCG to PCG32

Hashing scrambles a key once. A generator scrambles its own state over and over, and the simplest way to do that is a **linear congruential generator** (LCG), which steps $s_{t+1} = (A s_t + c) \bmod 2^{64}$. With $c$ odd and $A \equiv 1 \pmod 4$ it visits all $2^{64}$ states before repeating (the Hull-Dobell theorem). Its weakness is the low bits: bit $k$ of $s_t$ depends only on bits $0..k$ of the previous state, so it repeats with period $2^{k+1}$, and the lowest bit simply alternates. An LCG's raw low bits are not random at all.

PCG keeps the LCG for the state and outputs a **permutation** of the *old* state that uses only its good high bits:

```
next_u32():
    old   = state
    state = old * A + inc                          (mod 2^64)
    xs    = u32(((old >> 18) XOR old) >> 27)       (xorshift: high bits fold into the middle)
    rot   = old >> 59                              (the top 5 bits, 0 to 31)
    return rotr32(xs, rot)                         (a rotation chosen by the state itself)
```

Taking the output from `old` lets C compute the multiply and the permutation in parallel. `inc` selects one of $2^{63}$ **streams**: `inc = (seq << 1) | 1`, forced odd, because an even increment breaks the full period. **Seeding** is O'Neill's `pcg32_srandom_r(seed, seq)`: `state = 0`, one step, `state += seed`, one step. The two steps push the seed through the multiplier so that seeds 0 and 1 do not give nearly identical first outputs.

> **Warning:** In C, the rotation has one trap. `xs << (32 - rot)` with `rot = 0` shifts a 32-bit value by 32, which is **undefined behavior**: x86 and ARM happen to compute something, the optimizer may compute something else, and UBSan stops the program. Write the left shift as `xs << ((0u - rot) & 31u)`. For `rot = 0` it shifts by 0, and `xs | xs` is still `xs`.

## Uniform doubles from integers

The generator produces 32-bit integers, but most consumers want a number in $[0, 1)$. A double has a 53-bit significand, so the finest grid of evenly spaced doubles in $[0, 1)$ is $k \cdot 2^{-53}$ for $k = 0, \dots, 2^{53} - 1$. One 32-bit draw is not enough bits, so the spec uses two:

$$u = \left(\lfloor a / 2^5 \rfloor \cdot 2^{26} + \lfloor b / 2^6 \rfloor\right) \cdot 2^{-53},$$

taking 27 bits from $a$ and 26 from $b$. The integer in the parentheses is below $2^{53}$, so the double holds it exactly, and multiplying by a power of two is exact too. No rounding happens anywhere, which is why `uniform()` is bit-identical in every language, and why $u < 1$ always. Inverse-CDF sampling in [Sampling and logit processors](../../ai-engineering/21-inference/01-sampling-and-logit-processors.md) relies on never seeing 1.0. In C, compute in `uint64_t`: `(a << 26)` on a 32-bit type loses the top bits.

## Unbiased integers and shuffling

The other common request is an integer in $[0, n)$. Drawing it from $r$ uniform on $[0, 2^{32})$ as `r % n` is biased whenever $n$ does not divide $2^{32}$: the first $2^{32} \bmod n$ residues get one extra preimage. For $n = 3 \cdot 2^{30}$ the values below $2^{30}$ come up half the time instead of a third.

The fix is **rejection**. Let $t = 2^{32} \bmod n$, computed as `(2**32 - n) % n` so it fits in 32 bits; draw until $r \ge t$ and return $r \bmod n$. The accepted range has $2^{32} - t$ values, a multiple of $n$, so every residue is equally likely, and fewer than half of the draws are ever rejected.

With unbiased bounded integers you can shuffle. **Fisher-Yates** shuffles a list uniformly: for $i = n-1$ down to $1$, pick $j$ uniformly in $[0, i]$ and swap positions $i$ and $j$. Each of the $n!$ sequences of choices gives a different permutation, so all are equally likely. Picking $j$ in $[0, i)$ instead (Sattolo's algorithm) never leaves an element in place and produces only the $(n-1)!$ cyclic permutations.

## Sub-streams with SplitMix64

One user seed must feed several independent consumers (initialization, dropout, shuffling, sampling), so that turning dropout on does not change the initial weights. The spec derives each purpose's generator from the seed alone:

$$\mathrm{child\_seed}(s, p) = \mathrm{mix64}(s + p \cdot \texttt{0x9E3779B97F4A7C15}), \qquad \mathrm{stream}(s, \mathit{purpose}) = \mathrm{PCG32}(\mathrm{child\_seed}(s, p), \mathit{seq} = p),$$

with purpose ids init 1, dropout 2, shuffle 3, sample 4, mutation 5. `mix64` is SplitMix64's output function, two rounds of xorshift and odd multiply, both of them bijections as shown earlier:

```
z = (z XOR (z >> 30)) * 0xBF58476D1CE4E5B9
z = (z XOR (z >> 27)) * 0x94D049BB133111EB
return z XOR (z >> 31)                                          (all mod 2^64)
```

A SplitMix64 generator seeded with $s$ adds the golden-ratio constant to its state and mixes, so its $k$-th output is $\mathrm{mix64}(s + k \cdot \texttt{0x9E37...})$. The child seed of purpose $p$ is therefore simply output $p$ of SplitMix64 seeded with $s$. It depends on the seed, never on how many draws the parent made.

## Working it by hand

Start with the first output of `pcg32(0)`, the spec's `worked_example_seed_0`. Seeding with `seed = 0`, `seq = 54` goes like this:

| Step | Computation | Result |
|---|---|---|
| inc | $(54 \ll 1) \mid 1 = 108 + 1$ | 109 |
| first step | $0 \cdot A + 109$ | state = 109 |
| add the seed | $109 + 0$ | 109 |
| second step | $109 \cdot A + 109 \bmod 2^{64}$ | state = `0x9AE4F7499BA72696` |

Now call `next_u32()` with `old = 0x9AE4F7499BA72696`. The table follows the permutation one quantity at a time.

| Quantity | Value |
|---|---|
| `old >> 18` | `0x000026B93DD266E9` |
| `(old >> 18) XOR old` | `0x9AE4D1F0A675407F` |
| `... >> 27`, low 32 bits | `xs = 0x5C9A3E14` |
| `rot = old >> 59` | 19 (the top 5 bits, `10011`) |
| `rotr32(xs, 19)` = `(xs >> 19) \| (xs << 13)` (low 32 bits) | **`0x47C28B93`** = 1203932051 |

That is the first entry of `next_u32["0"]` in `spec/pcg32.vectors.json`. The second draw is `0xB98F6A27`, so the first `uniform()` uses $\lfloor \texttt{0x47C28B93} / 32 \rfloor = 37622876$ and $\lfloor \texttt{0xB98F6A27} / 64 \rfloor = 48643496$, giving $(37622876 \cdot 2^{26} + 48643496) \cdot 2^{-53} = 2524828517416360 \cdot 2^{-53} = 0.2803122753265841$.

FNV-1a of `"a"` is shorter. `"a"` is the byte `0x61`. $h = \texttt{0xCBF29CE484222325} \oplus \texttt{0x61} = \texttt{0xCBF29CE484222344}$, then $h \cdot \texttt{0x100000001B3} \bmod 2^{64} = \texttt{0xAF63DC4C8601EC8C}$. The full product is `0xCBF29CE5DEAF63DC4C8601EC8C`; keep the low 16 hex digits.

Finally, a universal hash with $a = 3$, $b = 7$, $p = 13$, $m = 5$, $x = 10$: $3 \cdot 10 + 7 = 37$; $37 \bmod 13 = 11$; $11 \bmod 5 = \mathbf{1}$.

## Designing the interface

The module exposes one generator class and four free functions:

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

`state` and `set_state` exist so a checkpoint can store the generator; `fnv1a64` takes an optional starting hash so that buffers chain.

## Building it step by step

The reasoning that leads to the code runs as follows.

1. Every random choice must replay from one seed in Python, Rust, and Go. So the generator cannot be a library's; it must be a fixed algorithm specified to the bit, with published vectors: PCG32 as in [`spec/pcg32.md`](../../../../course/contracts/spec/pcg32.md).
2. The state lives in $\mathbb{Z}_{2^{64}}$ and outputs in $\mathbb{Z}_{2^{32}}$. C gets the reduction from `uint64_t` overflow; Python ints never overflow, so every multiply and add that can exceed 64 bits is followed by `& MASK64`, and every 32-bit output by `& MASK32`.
3. The raw LCG has weak low bits, so the output is a permutation of the *old* state built from its high bits: an xorshift, then a rotation by the top 5 bits. The increment is forced odd, and seeding follows O'Neill's two-step procedure exactly, or no vector matches.
4. Anything built on top must also be bit-exact. A uniform double therefore uses only integer arithmetic and one exact scaling by $2^{-53}$; a bounded integer uses rejection, which is exact and unbiased; the shuffle is Fisher-Yates with $j \in [0, i]$.
5. Independent purposes need independent generators that depend only on the seed. So a sub-stream is a fresh `PCG32` seeded by `child_seed(seed, purpose)`, never derived from the parent's current state.
6. The hashes are the same arithmetic: FNV-1a is xor then multiply mod $2^{64}$, taking an optional starting value so buffers chain; the universal hash reduces mod $p$ before mod $m$.

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

Seeding is `pcg32_srandom_r(seed, seq)`: state 0, one step, add the seed, one step. The hand trace followed it for seed 0, landing on state `0x9AE4F7499BA72696`:

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

Last come the two hashes. `fnv1a64` takes the previous hash as its starting value, which is what makes `fnv1a64(b, fnv1a64(a)) == fnv1a64(a + b)`; `universal_hash` validates the family's parameters and reduces in the order that carries the $1/m$ guarantee:

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

The full solution is [`rng.py`](../../../../course/solutions/python/tinyllm/num/rng.py).

### Why this design

Start with cost. `next_u32` is a constant number of big-int operations on at most 128-bit values; `uniform` is two draws; `below` needs fewer than two draws on average, since the rejected fraction is $t / 2^{32} < 1/2$; `shuffle` is $O(n)$ draws; `fnv1a64` is $O(\text{bytes})$. Python makes all of this slow next to C, but training draws one number per weight or per mask element, which is fine at this scale, and the Rust engine ports the same code.

Why not numpy's or Python's generator? They are excellent generators, but their streams are not specified across languages, so a Python sampler and a Rust engine could never agree token for token, and a checkpoint could not store the generator in a portable form.

Why PCG32 rather than a raw LCG or xorshift? A raw LCG's low bits have tiny periods, and PCG hides them behind the permutation at almost no cost. xorshift generators fail some statistical tests on their low bits and have no cheap stream selection. PCG32 has small state (two words), streams, and published reference outputs.

Why rejection rather than `r % n` or floating-point scaling? `r % n` is biased. `floor(uniform() * n)` is also biased for large $n$ and depends on floating-point rounding. Rejection is exact and portable. Lemire's multiply-shift method is faster but would change the stream, and the spec fixes this one.

Why derive sub-streams from the seed? Deriving a child from the parent's current state would tie dropout to the number of initialization draws, so changing the model size would change the dropout masks.

One tradeoff is accepted: every arithmetic step carries an explicit mask. It is noise to read, but it is exactly the reduction modulo $2^{64}$ the mathematics requires, and dropping one mask is the bug that matches for a while and then diverges.

## What can go wrong

Most of the checks a correct solution must pass are published vectors, so a port in any language can be held to the same oracle. The table lists them with the downstream reason for each.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Seed 0 by hand | unit | the hand trace: the seeded state and `0x47C28B93` | pins the seeding procedure |
| O'Neill's demo line | golden | `PCG32(42)` gives O'Neill's published first six outputs | an oracle outside the books |
| 1024 outputs | golden | 1024 outputs for seeds 0, 1, $2^{63}$ | every language's stream |
| Bit-exact uniforms | golden | the spec's `uniform_f64` vectors, also through `uniforms` | the sampler's one draw per token |
| 53 bits, below 1 | property | $u \cdot 2^{53}$ integral, $u < 1$, more than 32 bits used | inverse-CDF sampling |
| `below` and `shuffle` vectors | golden | `below(10)` and `shuffle([0..9])` vectors | the training loop replays its data order on resume |
| No modulo bias | statistical | $n = 3 \cdot 2^{30}$: the share below $2^{30}$ is 1/3, not 1/2 | unbiased sampling of large ranges |
| Bad `n` for `below` | boundary | $n = 0$, negative, or above $2^{32}$ raise | |
| Sub-stream vectors | golden | `child_seed` and the first outputs of each purpose | init, dropout, shuffle, sample stay independent |
| Sub-stream independence | property | a sub-stream depends on the seed only | adding dropout does not change init |
| SplitMix64 outputs | golden | SplitMix64 seeded 0 and 1234567, published outputs | the mixer is the standard one |
| FNV-1a vectors | golden | FNV-1a 64 reference vectors | the KV block hash in the paged block pool and the Rust engine |
| FNV-1a chaining | property | `fnv(b, fnv(a)) == fnv(a + b)` on random bytes | chained block hashes |
| Universal hash by hand | unit | the hand-trace universal hash | |
| Universal collision rate | property | exhaustive over all $(a, b)$ for $p = 101$: collisions at most $1/m + 1/p$ | hash tables in *Algorithms* |
| State round trip | property | `set_state(state())` continues the stream; an even `inc` is rejected | checkpoints store the generator (`trainer_state.json`) |
| Negative `n` for `uniforms` | boundary | `uniforms(-1)` raises, `uniforms(0)` is empty | |

The pitfalls are mostly small departures from the spec that produce a plausible stream matching nothing else. Each is listed with its symptom and the line that prevents it.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| permuting the new state instead of the old one | a plausible stream that matches no other language | saves `old` before the step and permutes it |
| an even increment (`seq << 1` without `\| 1`) | short period; wrong from the first output | `inc = ((seq << 1) \| 1) & MASK64` |
| no `& MASK64` in Python | correct until the state first wraps, then wrong forever | masks after every multiply and add |
| a uniform from one 32-bit draw | only 32 random bits, so the lower 21 bits of a double are always zero | two draws, 27 + 26 bits, scaled by $2^{-53}$ |
| seeding in the wrong order (seed before the first step) | streams for nearby seeds look alike, and no vector matches | step, add the seed, step, as in `pcg32_srandom_r` |
| `j = below(i)` in the shuffle (Sattolo) | only cyclic permutations; no element ever stays put | `j = self.below(i + 1)` |
| `r % n` without rejection | small values favoured: 1/2 instead of 1/3 for $n = 3 \cdot 2^{30}$ | rejects `r < t` with `t = (2**32 - n) % n` |
| deriving a sub-stream from the current state | dropout masks change when the number of init draws changes | seeds the child from `self.seed`, never from `_state` |
| a different mixer (Murmur's `>> 33`) | child seeds differ from every other port | SplitMix64's shifts 30, 27, 31 and its two constants |
| multiply before xor (FNV-1, not FNV-1a) | every KV block hash differs from the Rust engine's | `h ^= byte` before `h *= FNV_PRIME` |
| reducing mod $m$ before mod $p$ | not universal; clustered buckets | `((a * x + b) % p) % m` |
| accepting an even `inc` in `set_state` | a corrupt checkpoint resumes a non-PCG sequence | `set_state` raises on an even `inc` |

> **Warning:** A missing mask in Python is the worst of these, because it raises nothing. The outputs match the vectors until the state first exceeds $2^{64}$, and then they are wrong forever. Test with at least a few hundred outputs, not just the first one.

## Where this leads

This generator is the single source of randomness for nearly everything that follows. In this book, [Random variables, expectation, variance, and normal draws by Box-Muller](../13-probability-statistics/01-random-variables-and-the-normal.md) builds `normal(rng, n)` from two `uniform()` draws.

In *AI Engineering*, the foundations chapter leans on it throughout. [The op library and gradcheck_all](../../ai-engineering/06-numerics-and-autograd/03-op-library-and-gradcheck.md) draws dropout masks and `gradcheck_all`'s random inputs from it. In [Module system and basic layers](../../ai-engineering/06-numerics-and-autograd/05-module-system-and-layers.md), layers default to `PCG32(0).substream(purpose)`, so initialization and dropout use separate streams and adding dropout does not change the initial weights. [Training loop and the autograd bigram](../../ai-engineering/06-numerics-and-autograd/06-training-loop-and-the-autograd-bigram.md) samples text with one `uniform()` per token. In [Safetensors for every dtype, atomic checkpoints, and the token stream](../../ai-engineering/06-numerics-and-autograd/07-safetensors-checkpoints-and-token-streams.md), `TokenStream` draws its window starts from a PCG32 and saves the generator's state in its cursor, so a resumed run reads the same batches.

Later, [Sampling and logit processors](../../ai-engineering/21-inference/01-sampling-and-logit-processors.md) draws one `uniform()` per token on `stream(seed, sample)` (`spec/sampling.md`), and the [Paged KV block pool](../../ai-engineering/21-inference/08-paged-kv-block-pool.md) chains FNV-1a block hashes. The Rust port in [Candle model runner, Rust KV cache, sampler and PCG32](../../ai-engineering/23-model-serving/02-model-runner-and-sampler.md) and the Go port in [An open-loop load generator](../../ai-engineering/23-model-serving/11-load-generator.md) are held to the same published vectors.

Almost every model section from [Bengio's neural probabilistic language model](../../ai-engineering/08-statistical-language-models/02-bengio-nplm.md) through the [Llama-family model](../../ai-engineering/13-decoder-block/09-llama-family-model.md) draws from this generator, including [word2vec](../../ai-engineering/08-statistical-language-models/03-word2vec-and-ppmi-svd.md), the [LSTM](../../ai-engineering/09-recurrent-networks/02-lstm.md), [GRU](../../ai-engineering/09-recurrent-networks/03-gru.md), and [stateful RNN language model](../../ai-engineering/09-recurrent-networks/06-rnn-language-model-stateful-tbptt.md), the attention-origins sections ([encoder-decoder with teacher forcing](../../ai-engineering/10-attention/01-encoder-decoder-teacher-forcing.md), [Bahdanau](../../ai-engineering/10-attention/02-bahdanau-additive-attention.md), [Luong](../../ai-engineering/10-attention/03-luong-attention-input-feeding.md)), the 2017 Transformer ([multi-head attention](../../ai-engineering/11-transformer/03-multi-head-attention.md), [positional encodings](../../ai-engineering/11-transformer/04-positional-encodings.md), [the encoder-decoder Transformer](../../ai-engineering/11-transformer/05-encoder-decoder-transformer.md)), the objectives chapter ([GPT](../../ai-engineering/12-objectives-and-adaptation/01-gpt-decoder-only.md), [BERT](../../ai-engineering/12-objectives-and-adaptation/02-bert-masked-lm.md), [ELECTRA](../../ai-engineering/12-objectives-and-adaptation/03-electra-replaced-token-detection.md), [fine-tuning heads](../../ai-engineering/12-objectives-and-adaptation/05-fine-tuning-heads-and-the-linear-head-export.md), [the evaluation harness](../../ai-engineering/12-objectives-and-adaptation/07-lm-evaluation-harness-and-the-model-zoo.md)), and the modern block ([gated MLPs](../../ai-engineering/13-decoder-block/02-gated-mlps.md), [GQA](../../ai-engineering/13-decoder-block/05-gqa-attention.md), [multi-head latent attention](../../ai-engineering/13-decoder-block/06-multi-head-latent-attention.md), [mixture of experts](../../ai-engineering/13-decoder-block/08-mixture-of-experts.md)).

The multimodal chapters use it too: conv weights in [Conv layers and pooling on autograd](../../ai-engineering/14-vision/01-conv-layers-and-pooling.md), patch, position, and block weights in the [Vision transformer encoder](../../ai-engineering/14-vision/03-vision-transformer.md), the dual encoder in [Contrastive image-text: CLIP and SigLIP](../../ai-engineering/14-vision/05-contrastive-image-text.md), [Image preprocessing to the specification](../../ai-engineering/14-vision/06-image-preprocessing.md), the [Whisper encoder](../../ai-engineering/15-speech/02-whisper-encoder.md), the [Whisper decoder and checkpoint weights](../../ai-engineering/15-speech/03-whisper-decoder-and-weights.md), [Whisper decoding](../../ai-engineering/15-speech/04-whisper-decoding.md), and `bootstrap_ci` in [Multimodal eval metrics](../../ai-engineering/15-speech/05-multimodal-eval-metrics.md), which draws its resamples from `PCG32(seed)`.

Outside that book, the Python Bloom screen in [Exact dedup: paragraph hashes, a Bloom screen, and a sort-merge confirm](../../data-engineering/05-corpus-pipeline/03-exact-dedup-bloom-and-sort-merge.md) in *Data Engineering* derives stable bit positions from FNV-1a and SplitMix64, [Near-duplicate dedup: MinHash, LSH, union-find, and decontamination](../../data-engineering/05-corpus-pipeline/04-near-dedup-minhash-lsh.md) builds on this generator too, and [Bias and safety evals](../../ai-engineering/36-bias-and-safety-evals/01-bias-and-safety-evals.md) in *AI Engineering* resamples paired prompts with PCG32 so its bootstrap intervals are reproducible.

Production systems take each piece further:

- The PCG family (`pcg64`, numpy's default `PCG64`) adds 128-bit state, the DXSM output function, and jump-ahead in $O(\log n)$; see numpy's `numpy/random/_pcg64.pyx` and O'Neill's `pcg-cpp`.
- JAX's counter-based `threefry` keys replace SplitMix64 sub-streams: a key splits into independent keys without any shared state, ideal for parallel devices; see `jax/_src/prng.py`.
- Lemire's nearly divisionless method replaces rejection in `below` with one multiply and rarely a division per draw.
- xxHash and wyhash replace FNV-1a with many bytes per step using SIMD and better avalanche, still non-cryptographic; see `xxhash.h` (`XXH3`).
- vLLM's prefix caching chains block hashes like ours, hashing each KV block with its parent hash and extra keys (LoRA id, multimodal inputs); see `hash_block_tokens` in `vllm/v1/core/kv_cache_utils.py`.

### Further reading

- O'Neill, "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation" (2014).
- Knuth, *The Art of Computer Programming*, vol. 2, ch. 3 (linear congruential generators, the spectral test).
- Steele, Lea, and Flood, "Fast Splittable Pseudorandom Number Generators" (2014).
- Carter and Wegman, "Universal Classes of Hash Functions" (1979).
- Lemire, "Fast Random Integer Generation in an Interval" (2019).
- Noll's FNV page (isthe.com/chongo/tech/comp/fnv).

## Key terms

- **Congruence**: $a \equiv b \pmod n$ when $a$ and $b$ have the same remainder mod $n$.
- **LCG**: a generator that steps $s_{t+1} = (A s_t + c) \bmod 2^{64}$.
- **PCG32**: an LCG whose output is a permutation (xorshift, then a state-chosen rotation) of the old state.
- **Stream**: one of $2^{63}$ sequences selected by the odd increment `inc`.
- **Rejection sampling**: discarding draws below $2^{32} \bmod n$ so that `r % n` is unbiased.
- **Fisher-Yates shuffle**: the uniform shuffle that swaps position $i$ with a uniform $j \in [0, i]$.
- **Universal family**: a random choice of hash function under which any two distinct keys collide with probability at most $1/m$.
- **FNV-1a**: a byte hash that xors each byte in and then multiplies by a fixed prime mod $2^{64}$.
- **SplitMix64**: the mixer used to derive child seeds that depend only on the seed and the purpose.

## Summary

- Python integers are unbounded, so PCG's 64-bit state transition explicitly applies `& (2**64 - 1)` after each multiply and add.
- PCG32 is a 64-bit LCG whose state is hidden behind a permutation chosen by its top bits; seeded the same way, Python, Rust, and Go produce the same stream bit for bit.
- A uniform double with all 53 bits takes two 32-bit draws and only integer arithmetic, so it is bit-exact across languages and never equals 1.0.
- `r mod n` is biased unless $n$ divides $2^{32}$; rejection removes the bias.
- Sub-streams derived with SplitMix64 keep initialization, dropout, shuffling, and sampling independent; FNV-1a chains over buffers, which is how a KV block hash extends its parent's.

## Review questions

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
