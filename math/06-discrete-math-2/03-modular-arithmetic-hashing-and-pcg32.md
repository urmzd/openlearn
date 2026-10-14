<!-- ol:module M06.3 -->
# Modular arithmetic, hashing, and PCG32 in Python

## Overview

| | |
|---|---|
| **Module** | `M06.3` · build · Python · Pass 2 · 4 to 5 h |
| **You build** | `python/tinyllm/num/rng.py`: `PCG32` (`next_u32`, `uniform`, `uniforms`, `below`, `shuffle`, `substream`, `state`, `set_state`), `splitmix64`, `child_seed`, `fnv1a64`, `universal_hash` |
| **Contract** | [`course/contracts/py/tinyllm/num/rng.pyi`](../../course/contracts/py/tinyllm/num/rng.pyi) · the algorithm, to the bit: [`spec/pcg32.md`](../../course/contracts/spec/pcg32.md) |
| **Tests** | `course/tests/M06.3/test_rng.py` checks Python against published golden vectors (section 4) |
| **Needs** | nothing to build first. Reading: `S-M05` (counting), `S-M06a` (modular arithmetic and hashing by hand) |
| **Used by** | `M07.0` normal draws by Box-Muller · `L0.2` dropout masks · `L0.4` the default init and dropout streams · `L0.5` the bigram's sampler · `L0.6` the token stream's window starts · later `L8.1` the sampler, `rt.04` KV block hashes, `data.03` Bloom hashing, `data.04` MinHash, and `ethics.04` reproducible bootstrap resampling; ported to Rust by `L10.1` and to Go by `load.01` · later: `L2.2`, `L2.3`, `L3.2`, `L3.3`, `L3.6`, `L4.1`, `L4.2`, `L4.3`, `L5.3`, `L5.4`, `L5.5`, `L6.1`, `L6.2`, `L6.3`, `L6.5`, `L6.7`, `L7.2`, `L7.5`, `L7.6`, `L7.8`, `L7.9` · `L14.5` the WER bootstrap stream (`PCG32(seed)`), `L13.1`, `L13.3`, `L13.5`, `L14.2`  · `L14.3` Whisper decoding · `L13.6` image preprocessing; `L14.4` |
| **Milestone** | `MS-P2` (the foundations gate) |
| **Optional depth** | O'Neill, "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation" (2014); Knuth, *TAOCP* vol. 2, ch. 3 (linear congruential generators, the spectral test); Steele, Lea, and Flood, "Fast Splittable Pseudorandom Number Generators" (2014); Carter and Wegman, "Universal Classes of Hash Functions" (1979); Noll's FNV page (isthe.com/chongo/tech/comp/fnv) |

## Key Takeaways

- Python integers are unbounded, so PCG's 64-bit state transition explicitly applies `& (2**64 - 1)` after each multiply and add (`test_first_1024_outputs_match_spec_vectors`).
- **PCG32** is a 64-bit linear congruential generator whose state is hidden behind a permutation (an xorshift and a rotation chosen by the top bits); seeded the same way, Python, Rust, and Go produce the same stream bit for bit (`test_first_1024_outputs_match_spec_vectors`).
- A **uniform double** with all 53 bits takes two 32-bit draws and only integer arithmetic, so it is bit-exact across languages and never equals 1.0 (`test_uniform_uses_53_bits_and_stays_below_one`).
- `r mod n` is **biased** unless $n$ divides $2^{32}$; rejection removes the bias (`test_below_has_no_modulo_bias`).
- **Sub-streams** derived with SplitMix64 keep initialization, dropout, shuffling, and sampling independent of each other (`test_substream_ignores_draws_already_made`); **FNV-1a** chains over buffers, which is how a KV block hash extends its parent's (`fnv1a64_chains_over_buffers`).

## How to work this chapter

```bash
ol start M06.3              # stubs python/tinyllm/num/rng.py
ol tests M06.3              # read the test catalog first: rung R0, you write no tests here
ol check M06.3              # Python tests against published golden vectors
ol diff  M06.3              # after passing: your code against the reference
```

---

## 1. Why now

From Pass 2 on, your system makes random choices everywhere: initial weights, dropout masks, the order of training windows, the token the sampler picks. Every one of them has to be reproducible from one seed (P11), or a failing run cannot be replayed, a resumed checkpoint diverges from the run it continues, and a parity test between your Python sampler and your Rust engine (`L10.1`) has nothing to compare. numpy's generator, Rust's `rand`, and Go's `math/rand` all produce different streams for the same seed. So the course fixes one generator to the bit, PCG32 (`spec/pcg32.md`), and you implement it in Python for training. Published vectors give the later Rust and Go ports the same oracle. The same mathematics, arithmetic modulo a power of two, gives you the hash functions the system needs: FNV-1a for KV block hashes and universal hashing for hash tables.

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

Every integer $a$ and modulus $n \ge 1$ give a unique quotient and remainder, $a = qn + r$ with $0 \le r < n$; write $r = a \bmod n$. Two integers are **congruent** mod $n$ when they have the same remainder. Remainders respect addition and multiplication: $(a + b) \bmod n = ((a \bmod n) + (b \bmod n)) \bmod n$, and the same for $\times$. So you may reduce at every step instead of at the end, and $\mathbb{Z}_n$ is a closed number system.

A `uint64_t` in C holds exactly $\mathbb{Z}_{2^{64}}$: the C standard defines unsigned overflow as wraparound, which *is* reduction mod $2^{64}$. Python integers never overflow, so `x * y` of two 64-bit values has up to 128 bits; you reduce with `& (2**64 - 1)` (keeping the low 64 bits is the remainder mod $2^{64}$). Forget it once and the state grows without bound while the outputs, which read the low bits, go quietly wrong after the first wrap.

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

**The first output of `pcg32(0)`** (`test_worked_example_first_output_of_seed_0`, `worked_example_seed_0`). Seeding with `seed = 0`, `seq = 54`:

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

That is the first entry of `next_u32["0"]` in `spec/pcg32.vectors.json`. The second draw is `0xB98F6A27`, so the first `uniform()` is $\lfloor \texttt{0x47C28B93} / 32 \rfloor = 37622876$ and $\lfloor \texttt{0xB98F6A27} / 64 \rfloor = 48643496$, giving $(37622876 \cdot 2^{26} + 48643496) \cdot 2^{-53} = 2524828517416360 \cdot 2^{-53} = 0.2803122753265841$ (`test_uniform_bit_exact`).

**FNV-1a of `"a"`** (`test_fnv1a64_published_vectors`): `"a"` is the byte `0x61`. $h = \texttt{0xCBF29CE484222325} \oplus \texttt{0x61} = \texttt{0xCBF29CE484222344}$, then $h \cdot \texttt{0x100000001B3} \bmod 2^{64} = \texttt{0xAF63DC4C8601EC8C}$ (the full product is `0xCBF29CE5DEAF63DC4C8601EC8C`; keep the low 16 hex digits).

**A universal hash** (`test_universal_hash_hand_example`): $a = 3$, $b = 7$, $p = 13$, $m = 5$, $x = 10$. $3 \cdot 10 + 7 = 37$; $37 \bmod 13 = 11$; $11 \bmod 5 = \mathbf{1}$.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_worked_example_first_output_of_seed_0` | unit, smoke | section 3: the seeded state and `0x47C28B93` | you and the spec agree on seeding |
| `test_oneill_demo_line` | golden | `PCG32(42)` gives O'Neill's published first six outputs | an oracle outside the course |
| `test_first_1024_outputs_match_spec_vectors` | golden | 1024 outputs for seeds 0, 1, $2^{63}$ | every language's stream (D10) |
| `test_uniform_bit_exact` | golden | the spec's `uniform_f64` vectors, also through `uniforms` | the sampler's one draw per token |
| `test_uniform_uses_53_bits_and_stays_below_one` | property | $u \cdot 2^{53}$ integral, $u < 1$, more than 32 bits used | inverse-CDF sampling in `L8.1` |
| `test_below_and_shuffle_match_spec_vectors` | golden | `below(10)` and `shuffle([0..9])` vectors | `L0.5` replays its data order on resume |
| `test_below_has_no_modulo_bias` | statistical | $n = 3 \cdot 2^{30}$: the share below $2^{30}$ is 1/3, not 1/2 | unbiased sampling of large ranges |
| `test_below_rejects_bad_n` | boundary | $n = 0$, negative, or above $2^{32}$ raise | |
| `test_substreams_match_spec_vectors` | golden | `child_seed` and the first outputs of each purpose | init, dropout, shuffle, sample stay independent |
| `test_substream_ignores_draws_already_made` | property | a sub-stream depends on the seed only | adding dropout does not change init |
| `test_splitmix64_published_outputs` | golden | SplitMix64 seeded 0 and 1234567, published outputs | the mixer is the standard one |
| `test_fnv1a64_published_vectors` | golden, smoke | FNV-1a 64 reference vectors | the KV block hash in `rt.04` and the Rust engine |
| `test_fnv1a64_chains` | property | `fnv(b, fnv(a)) == fnv(a + b)` on random bytes | chained block hashes |
| `test_universal_hash_hand_example` | unit | section 3's universal hash | |
| `test_universal_hash_collision_rate` | property | exhaustive over all $(a, b)$ for $p = 101$: collisions at most $1/m + 1/p$ | hash tables in `ds.*` |
| `test_state_roundtrip_resumes_the_stream` | property | `set_state(state())` continues the stream; an even `inc` is rejected | checkpoints store the generator (`trainer_state.json`) |
| `test_uniforms_rejects_negative_n` | boundary | `uniforms(-1)` raises, `uniforms(0)` is empty | |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| permuting the new state instead of the old one | a plausible stream that matches no other language | `test_first_1024_outputs_match_spec_vectors` (mutant `s01`) |
| an even increment (`seq << 1` without `\| 1`) | short period; wrong from the first output | `test_worked_example_first_output_of_seed_0` (mutant `s02`) |
| no `& MASK64` in Python | correct until the state first wraps, then wrong forever | `test_first_1024_outputs_match_spec_vectors` (mutant `s03`) |
| a uniform from one 32-bit draw | only 32 random bits, so the lower 21 bits of a double are always zero | `test_uniform_uses_53_bits_and_stays_below_one` (mutant `s04`) |
| seeding in the wrong order (seed before the first step) | streams for nearby seeds look alike, and no vector matches | `test_oneill_demo_line` (mutant `s05`) |
| `j = below(i)` in the shuffle (Sattolo) | only cyclic permutations; no element ever stays put | `test_below_and_shuffle_match_spec_vectors` (mutant `s06`) |
| `r % n` without rejection | small values favoured: 1/2 instead of 1/3 for $n = 3 \cdot 2^{30}$ | `test_below_has_no_modulo_bias` (mutant `s07`) |
| deriving a sub-stream from the current state | dropout masks change when the number of init draws changes | `test_substream_ignores_draws_already_made` (mutant `s08`) |
| a different mixer (Murmur's `>> 33`) | child seeds differ from every other port | `test_splitmix64_published_outputs` (mutant `s09`) |
| multiply before xor (FNV-1, not FNV-1a) | every KV block hash differs from the Rust engine's | `test_fnv1a64_published_vectors` (mutant `s10`) |
| reducing mod $m$ before mod $p$ | not universal; clustered buckets | `test_universal_hash_hand_example` (mutant `s11`) |
| accepting an even `inc` in `set_state` | a corrupt checkpoint resumes a non-PCG sequence | `test_state_roundtrip_resumes_the_stream` (mutant `s12`) |

## 6. Where it's used next
| Forward | `L14.3` | Whisper decoding calls this module through its declared contract. |
| Forward | `L13.6` | Image preprocessing uses this module directly. |
| Forward | `L2.2` | Registered call site uses this module. |
| Forward | `L2.3` | Registered call site uses this module. |
| Forward | `L3.2` | Registered call site uses this module. |
| Forward | `L3.3` | Registered call site uses this module. |
| Forward | `L3.6` | Registered call site uses this module. |
| Forward | `L4.1` | Registered call site uses this module. |
| Forward | `L4.2` | Registered call site uses this module. |
| Forward | `L4.3` | Registered call site uses this module. |
| Forward | `L5.3` | Registered call site uses this module. |
| Forward | `L5.4` | Registered call site uses this module. |
| Forward | `L5.5` | Registered call site uses this module. |
| Forward | `L6.1` | Registered call site uses this module. |
| Forward | `L6.2` | Registered call site uses this module. |
| Forward | `L6.3` | Registered call site uses this module. |
| Forward | `L6.5` | Registered call site uses this module. |
| Forward | `L6.7` | Registered call site uses this module. |
| Forward | `L7.2` | Registered call site uses this module. |
| Forward | `L7.5` | Registered call site uses this module. |
| Forward | `L7.6` | Registered call site uses this module. |
| Forward | `L7.8` | Registered call site uses this module. |
| Forward | `L7.9` | Registered call site uses this module. |
| Forward | `data.04` | Registered call site uses this module. |

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

| Forward | `L14.4` | Whisper decoding consumes this implementation. |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| PCG32 | the PCG family (`pcg64`, numpy's default `PCG64`) | 128-bit state, the DXSM output function, jump-ahead in $O(\log n)$ | numpy `numpy/random/_pcg64.pyx`; O'Neill's `pcg-cpp` |
| sub-streams by SplitMix64 | JAX's counter-based `threefry` keys | split a key into independent keys without any shared state, ideal for parallel devices | `jax/_src/prng.py` |
| `below` by rejection | Lemire's nearly divisionless method | one multiply and rarely a division instead of a modulo per draw | Lemire, "Fast Random Integer Generation in an Interval" (2019) |
| FNV-1a | xxHash, wyhash | many bytes per step with SIMD, better avalanche, still non-cryptographic | `xxhash.h` (`XXH3`) |
| block hash chaining | vLLM prefix caching | hashes each KV block with its parent hash and extra keys (LoRA id, multimodal inputs) | vLLM `vllm/v1/core/kv_cache_utils.py`, `hash_block_tokens` |

